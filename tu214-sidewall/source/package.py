"""Package only the current, validated CAD revision and reproducible sources."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
CAD=ROOT/"cad"
SOURCE_HASH="370fe795332fc1ff231cc2dd068df0051266bb9e5ab8a86258c18059c9a0160d"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def check_current_revision():
    assert sha(CAD/"tu214_sidewall_original.step")==SOURCE_HASH,"Original panel changed"
    build=read(CAD/"build_validation.json");geometry=read(CAD/"geometry_validation.json")
    assert build["parameters"]==read(ROOT/"source/parameters.json"),"Rebuild after parameter changes"
    assert build["source_sha256"]==geometry["source_sha256"]==SOURCE_HASH
    assert geometry["all_geometry_checks_passed"]
    assert len(build["positions"])==len(geometry["assemblies"])==3
    for state in build["positions"]:
        assert state["valid"] and state["solids"]==11
        assert state["original_geometry_values_unchanged"] and state["panel_volume_difference_mm3"]==0
    for state in geometry["assemblies"]:
        assert state["geometry_hash_matches_original"]
        assert sha(CAD/state["file"])==state["sha256"],"Assembly changed since geometry validation"
    for index in (1,2):
        contact=read(CAD/f"collision_window_{index}.json")
        assert contact["window"]==index
        assert contact["continuous_motion_checked_by_full_swept_envelopes"]
        assert contact["reveal_vs_shade_handle_guides_certified_by_radial_control_hull"]
        assert contact["handle_vs_guides_certified_by_disjoint_x_ranges"]
        assert len(contact["checks"])==7
        assert all(c["intersection_valid"] and c["intersection_mm3"]<.001 for c in contact["checks"])
        for name,digest in contact["part_sha256"].items():
            assert sha(CAD/f"{name}.step")==digest,"Rerun contacts for the current components"
    newest_cad=max(p.stat().st_mtime for p in CAD.glob("*.step") if not p.name.startswith("motion_envelope_"))
    for name in ("front","perspective","handle_detail","shade_half_closed","vertical_section"):
        path=ROOT/"previews"/f"{name}.png"
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_mtime>=newest_cad,"Rerender after rebuilding CAD"
    return geometry


def deliverable_files():
    files=[ROOT/"README.md",ROOT/"requirements.txt",ROOT/".gitattributes"]
    for directory in ("cad","source","previews"):
        files.extend(p for p in (ROOT/directory).iterdir() if p.is_file())
    files.append(ROOT/"references/18205_marked.png")
    return sorted(files,key=lambda p:p.relative_to(ROOT).as_posix())


def main():
    geometry=check_current_revision();files=deliverable_files()
    manifest={"original_sha256":SOURCE_HASH,"all_geometry_checks_passed":True,"collision_checks_passed":True,
        "panel_preserved_in_all_three_assemblies":True,"assemblies_solid_count":11,
        "files":{p.relative_to(ROOT).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p)} for p in files},
        "geometry_report_sha256":sha(CAD/"geometry_validation.json")}
    manifest_path=ROOT/"manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    main_archive=ROOT/"tu214_sidewall_complete_STP.zip"
    with zipfile.ZipFile(main_archive,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.write(CAD/"tu214_sidewall_complete.step","tu214_sidewall_complete.stp")
    archive=ROOT/"tu214_sidewall_complete_package.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in files+[manifest_path]:z.write(path,"tu214-sidewall/"+path.relative_to(ROOT).as_posix())
    for path in (main_archive,archive):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
    with zipfile.ZipFile(main_archive) as z:
        assert hashlib.sha256(z.read("tu214_sidewall_complete.stp")).hexdigest()==sha(CAD/"tu214_sidewall_complete.step")
    with zipfile.ZipFile(archive) as z:
        for path in files:
            assert hashlib.sha256(z.read("tu214-sidewall/"+path.relative_to(ROOT).as_posix())).hexdigest()==sha(path)
    print(json.dumps({"files":len(files),"original_preserved":True,"assembly_solids":11,
        "archives":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in (main_archive,archive)}},indent=2))


if __name__=="__main__":main()
