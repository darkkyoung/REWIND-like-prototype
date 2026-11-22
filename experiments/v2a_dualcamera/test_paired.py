from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import test_adapter as fixture
from .adapter import PINS, mixed_stages, normalized_binding
from .decision import decide


class BindingTests(unittest.TestCase):
    def test_weight_roundtrip_preserves_pin_and_order(self):
        pin = deepcopy(PINS["root_binding"])
        self.assertEqual(normalized_binding(pin), normalized_binding({**pin,"weight":tuple(pin["weight"])}))
        self.assertIsInstance(pin["weight"],list)

    def test_reject_malformed_weights(self):
        for w in ("12",{},[1],[1,2,3],[True,0.1],["0.1",0.2],[float("inf"),0],[float("nan"),0]):
            with self.subTest(weight=w), self.assertRaises(ValueError):
                normalized_binding({"weight":w})

    def test_loader_rejects_changed_weight_and_binding_metadata(self):
        pin = PINS["root_binding"]
        changes = ({"weight":pin["weight"][::-1]},{"weight":[pin["weight"][0]+1e-15,pin["weight"][1]]},
                   {"parent":"other"},{"selection":"other"},{"implementation":{}},{"source":"extra"})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"fixture.pt"
            for change in changes:
                torch.save({"step":24,"binding":{**pin,**change}},path)
                with patch.dict(PINS,anchored=hashlib.sha256(path.read_bytes()).hexdigest()):
                    with self.subTest(change=change), self.assertRaisesRegex(ValueError,"experiment binding"):
                        mixed_stages(None,None,path)

    def test_step_and_file_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"fixture.pt"
            torch.save({"step":23,"binding":PINS["root_binding"]},path)
            with self.assertRaisesRegex(ValueError,"SHA"):
                mixed_stages(None,None,path)
            with patch.dict(PINS,anchored=hashlib.sha256(path.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError,"binding"):
                    mixed_stages(None,None,path)

    def test_real_architecture_tuple_checkpoint_strict_load_and_readback(self):
        env = fixture.CascadeTests
        env.setUpClass()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"fixture.pt"
            pin = PINS["root_binding"]
            torch.save({"binding":{**pin,"weight":tuple(pin["weight"])},"step":24,
                        "body":env.body.model.state_dict()},path)
            with path.open("rb") as f:
                sha = hashlib.file_digest(f,"sha256").hexdigest()
            with patch.dict(PINS,anchored=sha):
                body,hand = mixed_stages(env.st,env.hand,path)
                self.assertEqual(body.checkpoint_sha256,sha)
                self.assertEqual(len(body.model.state_dict()),104)
                for loaded,original in ((body,env.body),(hand,env.hand)):
                    for k,x in loaded.model.state_dict().items():
                        self.assertTrue(torch.equal(x,original.model.state_dict()[k]))
            with path.open("rb") as f:
                self.assertEqual(hashlib.file_digest(f,"sha256").hexdigest(),sha)


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.rows = {k:dict(sequence=s,frames=list(range(f,f+50)),same_environment_xy_exact=True,
                            same_environment_whole62_exact=True,same_environment_local_hand_exact=True,
                            published_parity={"status":"within_tolerance"}) for k,(s,f) in PINS["clips"].items()}
        self.first = next(iter(self.rows.values()))

    def test_exploratory_preserves_release_fail(self):
        self.first["published_parity"]["status"] = "outside_tolerance"
        for mode,allow in (("strict",False),("exploratory",True)):
            v = decide(mode,self.rows)
            self.assertEqual(v["strict_release"],"FAIL")
            self.assertEqual(v["parity_allows_candidate"],allow)

    def test_same_device_failure_blocks_both(self):
        for key in ("same_environment_xy_exact","same_environment_whole62_exact","same_environment_local_hand_exact"):
            self.first[key] = False
            for mode in ("strict","exploratory"):
                self.assertFalse(decide(mode,self.rows)["parity_allows_candidate"])
            self.first[key] = True

    def test_complete_fixed_selection_required(self):
        with self.assertRaises(ValueError):
            decide("exploratory",{})
        self.first["frames"][0] += 1
        with self.assertRaises(ValueError):
            decide("exploratory",self.rows)

    def test_unrecognized_status_fails_closed(self):
        self.first["published_parity"]["status"] = "not_run"
        with self.assertRaises(ValueError):
            decide("exploratory",self.rows)

    def test_full_pass_retains_historical_verdict(self):
        v = decide("strict",self.rows)
        self.assertTrue(v["parity_allows_candidate"])
        self.assertEqual((v["historical_release"],v["strict_release"]),("FAIL","PASS"))


class SplitAuditTests(unittest.TestCase):
    def setUp(self):
        from .run import split_audit
        self.audit = split_audit
        self.selected = [("scene", ("validation_person/walk", 10))]
        def row(split, sequence, frame):
            files = {role: {"sha256": hashlib.sha256(f"{sequence}/{frame}/{role}".encode()).hexdigest()}
                     for role in ("left", "right", "pose")}
            return {"split": split, "sequence_id": sequence, "frame_index": frame, "files": files}
        self.train = row("train", "train_person/walk", 10)
        self.validation = [row("validation", "validation_person/walk", f) for f in range(10, 60)]

    def test_selected_frames_and_distinct_train_files(self):
        result = self.audit([self.train, *self.validation], self.selected)
        self.assertEqual(result["frames"], 50)
        self.assertEqual(result["exact_train_file_overlap"], {"left": 0, "right": 0, "pose": 0})
        self.assertEqual(result["train_motion_name_overlap"], ["walk"])

    def test_selected_frame_from_train_is_rejected(self):
        self.validation[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "validation membership"):
            self.audit([self.train, *self.validation], self.selected)

    def test_missing_or_duplicate_frame_is_rejected(self):
        for rows in (self.validation[1:], self.validation[1:] + [self.validation[-1]]):
            with self.subTest(), self.assertRaisesRegex(ValueError, "validation membership"):
                self.audit([self.train, *rows], self.selected)

    def test_sha_overlap_is_rejected_for_each_file_role(self):
        for role in self.train["files"]:
            rows = deepcopy(self.validation)
            rows[-1]["files"][role] = self.train["files"][role]
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "exact train file overlap"):
                self.audit([self.train, *rows], self.selected)
