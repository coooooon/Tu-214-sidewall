"""Keep the supplied panel's original STEP B-rep records in the assembly.

An ordinary import/export can reparameterize trim curves. Copy the original
solid's dependency graph instead, changing only entity numbers. The new
assembly retains its product structure, units and accessory geometry.
"""
import re
from pathlib import Path

ENTITY=re.compile(r"#(\d+)\s*=\s*((?:'(?:[^']|'')*'|[^';])*);",re.S)
TOKEN=re.compile(r"'(?:[^']|'')*'|#(\d+)")


def references(record):
    return {int(m.group(1)) for m in TOKEN.finditer(record) if m.group(1)}


def records(text):
    data=text.split("DATA;",1)[1].split("ENDSEC;",1)[0]
    return {int(m.group(1)):m.group(2) for m in ENTITY.finditer(data)}


def closure(items,seeds):
    found=set();queue=list(seeds)
    while queue:
        i=queue.pop()
        if i in found:continue
        if i not in items:raise ValueError(f"Undefined STEP reference #{i}")
        found.add(i);queue.extend(references(items[i])-found)
    return found


def preserve_panel(source_path,assembly_path,panel_name="Wall_Panel"):
    source_text=Path(source_path).read_text();assembly_text=Path(assembly_path).read_text()
    original=records(source_text);items=records(assembly_text)
    originals=[i for i,r in original.items() if r.lstrip().startswith("MANIFOLD_SOLID_BREP(")]
    if len(originals)!=1:raise ValueError("Expected one solid in the supplied panel STEP")
    panel_product=next(i for i,r in items.items() if r.startswith(f"PRODUCT('{panel_name}'"))
    formation=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION_FORMATION(") and panel_product in references(r))
    definition=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION(") and formation in references(r))
    product_shape=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION_SHAPE(") and definition in references(r))
    shape_definition=next(r for r in items.values() if r.startswith("SHAPE_DEFINITION_REPRESENTATION(") and product_shape in references(r))
    representation=next(i for i in references(shape_definition) if items[i].startswith("ADVANCED_BREP_SHAPE_REPRESENTATION("))
    target=next(i for i in references(items[representation]) if items[i].startswith("MANIFOLD_SOLID_BREP("))
    original_root=originals[0]
    original_representation=next(i for i,r in original.items() if r.lstrip().startswith("ADVANCED_BREP_SHAPE_REPRESENTATION(") and original_root in references(r))
    old=closure(items,[representation]);old_without_root=old-{target,representation}
    # Preserve shared dependencies used by other geometry/assembly entities.
    external=set().union(*(references(r) for i,r in items.items() if i not in old))&old_without_root
    retained=closure(items,external) if external else set()
    for i in old_without_root-retained:del items[i]
    original_geometry=closure(original,[original_representation])
    offset=max(items)+1
    mapping={i:offset+i for i in original_geometry};mapping[original_root]=target;mapping[original_representation]=representation
    for i in original_geometry:
        items[mapping[i]]=TOKEN.sub(lambda m:f"#{mapping[int(m.group(1))]}" if m.group(1) else m.group(0),original[i])
    unresolved=set().union(*(references(r) for r in items.values()))-set(items)
    if unresolved:raise ValueError(f"Undefined merged STEP references: {unresolved}")
    header=assembly_text.split("DATA;",1)[0]+"DATA;\n"
    output=header+"\n".join(f"#{i} = {r};" for i,r in sorted(items.items()))+"\nENDSEC;\nEND-ISO-10303-21;\n"
    Path(assembly_path).write_text(output)
    # Numerical values and entity types must be byte-identical after mapping
    # the new reference numbers back to the supplied original numbers.
    reverse={v:k for k,v in mapping.items()}
    for i in original_geometry:
        restored=TOKEN.sub(lambda m:f"#{reverse[int(m.group(1))]}" if m.group(1) else m.group(0),items[mapping[i]])
        if restored!=original[i]:raise ValueError("Original STEP geometry record changed")
    return {"original_geometry_records_preserved":len(original_geometry),"original_geometry_values_unchanged":True}


if __name__=="__main__":
    import sys
    print(preserve_panel(sys.argv[1],sys.argv[2]))
