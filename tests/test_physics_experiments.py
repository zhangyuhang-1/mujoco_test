"""Physical invariants and portable recording tests; no display or encoder required."""
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from dualstack.experiments.registry import cases
from dualstack.experiments.scenes import scene, initialize
from dualstack.experiments.physics import run, COLUMNS, metrics
from dualstack.experiments.artifacts import package, zip_package
from dualstack.experiments.replay_standalone import validate


def test_effective_friction_and_energy_dissipation(tmp_path):
    distances = []
    for config in cases("01_friction"):
        result = run(config, tmp_path / config["name"])
        assert result["effective_mu_range"] == pytest.approx([config["mu"],config["mu"]])
        assert result["outcome"] == "stopped"
        assert result["max_tilt_rad"] < .1
        assert result["final_kinetic_energy_J"] < result["initial_kinetic_energy_J"] * .001
        assert abs(result["ideal_relative_error"]) < .08
        distances.append(result["sliding_distance_m"])
    assert distances[0] > distances[1] > distances[2]


def test_bounce_stiffness_is_independent_and_damping_is_not_restitution(tmp_path):
    heights = []
    for config in cases("02_bounce"):
        root = scene(config)
        pair = root.find("contact/pair[@geom1='sample_geom']")
        assert list(map(float, pair.get("solref").split())) == [-config["contact_stiffness"], -config["contact_damping"]]
        result = run(config, tmp_path / config["name"], record=False)
        assert result["first_impact_time_s"] is not None
        assert result["max_penetration_m"] < .0025
        assert not result["warnings"]
        heights.append(result["first_rebound_height_m"])
    assert heights[0] > heights[-1]


def test_button_equilibrium_and_release_force(tmp_path):
    config = dict(cases("04_button")[1], press_mode="force", release_time_s=1.)
    m = mujoco.MjModel.from_xml_string(ET.tostring(scene(config), encoding="unicode"))
    d = mujoco.MjData(m); initialize(m,d,config)
    j = m.joint("sample_joint")
    assert d.qpos[j.qposadr[0]] == pytest.approx(-.981/config["stiffness"])
    result = run(config,tmp_path)
    a = np.genfromtxt(tmp_path / "data.csv", delimiter=",", names=True)
    assert np.all(a["applied_force_N"][a["time_s"]>=1.00001] == 0)
    np.testing.assert_allclose(a["spring_force_N"],-config["stiffness"]*a["joint_q_m"])
    np.testing.assert_allclose(a["damper_force_N"],-config["damping"]*a["joint_v_m_s"])
    assert abs(result["final_equilibrium_error_m"]) < .0001


def test_classification_uses_observed_motion_not_offset(tmp_path):
    config=cases("03_stack")[0]
    result=run(config,tmp_path,record=False)
    a=np.genfromtxt(tmp_path / "data.csv",delimiter=",",skip_header=1)
    assert result["outcome"]=="stable"
    # Changing input label/offset alone cannot turn a recorded stable trajectory into a fall.
    assert metrics(a,dict(config,offset_m=.055))["outcome"]=="stable"


def test_replay_export_is_relative_and_loads_without_source_tree(tmp_path):
    config=cases("01_friction")[0]
    path=tmp_path / "run"
    result=run(config,path)
    root=package("01_friction",[dict(name=config["name"],path=path,config=config,metrics=result)],tmp_path / "export")
    archive=tmp_path / "portable.zip"; zip_package(root,archive)
    import zipfile
    moved=tmp_path / "moved"
    with zipfile.ZipFile(archive) as z: z.extractall(moved)
    for file in [moved / "scene.xml", *list((moved / "scenes").glob("*.xml"))]:
        for el in ET.parse(file).iter():
            if "file" in el.attrib:
                assert not __import__("pathlib").Path(el.get("file")).is_absolute()
                assert (file.parent / el.get("file")).exists()
    report=validate(moved)
    assert report["status"]=="passed"
    assert report["groups"][0]["max_pose_error"]<1e-10
    replay=(moved / "replay.py").read_text()
    assert "mujoco.mj_step(" not in replay


def test_video_fallback_is_bounded_and_manifest_serializable(tmp_path, monkeypatch):
    from dualstack.experiments import runner
    attempts=[]
    def simulated_child(command, logfile, timeout, backend):
        width=int(command[command.index("--width")+1])
        attempts.append(width)
        return dict(status="passed" if width==1280 else "failed", reason="injected test failure" if width!=1280 else None)
    monkeypatch.setattr(runner,"child",simulated_child)
    result=runner.render_with_fallback(tmp_path,tmp_path,"overview",30,10)
    assert attempts==[1920,1280]
    assert result["degraded"] and result["status"]=="passed"
    # In particular, the attempt list cannot contain a reference to its parent result.
    assert json.loads(json.dumps(result))["attempts"][0]["status"]=="failed"


def test_robot_release_is_measured_from_contact_not_waypoint_clock(tmp_path):
    config=dict(cases("04_button")[1],press_mode="robot",duration=4.5,release_time_s=3.3)
    result=run(config,tmp_path)
    assert result["max_depression_m"]>.005
    assert 2.7<result["actual_release_time_s"]<3.3
    a=np.genfromtxt(tmp_path / "data.csv",delimiter=",",names=True)
    assert not np.any(a["normal_force_N"][a["time_s"]>=result["actual_release_time_s"]]>1e-5)
    assert result["peak_contact_force_N"]>0
