"""
Finalize the flood inventory dataset starting from the raw merged
file: repair geometry, dissolve overlaps across events, remove
slivers, compute recurrence count/category and area, and arrange
a clear attribute schema for training point generation.

Input:  FloodPoints_AllEvents_Merged (raw merge of all 5 events)
Output: FloodInventory_Final_Clean -- one clean feature class with
        a well-organized attribute table
"""

import arcpy

arcpy.env.workspace = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Flash_Flood_Makkah.gdb"
arcpy.env.overwriteOutput = True

input_fc = "FloodPoints_AllEvents_Merged"
repaired_fc = "FloodPoints_AllEvents_Repaired"
dissolved_fc = "FloodExtent_AllEvents_Dissolved"
output_fc = "FloodInventory_Final"

# ------------------------------------------------------------
# 1. REPAIR GEOMETRY ON THE RAW MERGE
# ------------------------------------------------------------
print("Repairing geometry on merged input...")
arcpy.management.CopyFeatures(input_fc, repaired_fc)
arcpy.management.RepairGeometry(repaired_fc, delete_null="DELETE_NULL")

# ------------------------------------------------------------
# 2. DISSOLVE OVERLAPS ACROSS EVENTS
#    Prevents double-sampling zones that flooded in more than
#    one event when you generate training points later.
# ------------------------------------------------------------
print("Dissolving overlaps across events...")
arcpy.management.Dissolve(
    repaired_fc, dissolved_fc,
    dissolve_field=None,
    statistics_fields=[["event_date", "CONCATENATE"]],
    multi_part="SINGLE_PART",
    concatenation_separator=","
)

# ------------------------------------------------------------
# 3. START THE FINAL SCHEMA FROM THE DISSOLVED LAYER
# ------------------------------------------------------------
print("Copying dissolved layer as the base for the final inventory...")
arcpy.management.CopyFeatures(dissolved_fc, output_fc)

# ------------------------------------------------------------
# 4. REMOVE SLIVER POLYGONS
#    Anything under 500 sq m is almost certainly leftover noise
#    (dissolve can reintroduce tiny fragments at polygon boundaries)
# ------------------------------------------------------------
print("Computing area and removing slivers...")
arcpy.management.AddField(output_fc, "area_m2", "DOUBLE")
arcpy.management.CalculateGeometryAttributes(output_fc, [["area_m2", "AREA_GEODESIC"]], area_unit="SQUARE_METERS")

with arcpy.da.UpdateCursor(output_fc, ["area_m2"]) as cursor:
    removed = 0
    for row in cursor:
        if row[0] is None or row[0] < 500:
            cursor.deleteRow()
            removed += 1
print(f"Removed {removed} sliver polygons under 500 sq m")

# ------------------------------------------------------------
# 5. FIND THE CONCATENATED EVENT FIELD AND RENAME IT
#    (ArcGIS's exact naming for CONCATENATE statistics fields
#    can vary slightly by version -- check the printed list
#    below if this doesn't find it automatically)
# ------------------------------------------------------------
existing_fields = [f.name for f in arcpy.ListFields(output_fc)]
print("Fields found on input:", existing_fields)

concat_field = next((f for f in existing_fields
                      if "event_date" in f.lower() and "concat" in f.lower()), None)

if concat_field:
    arcpy.management.AlterField(output_fc, concat_field, "flood_events", "flood_events")
    events_field_name = "flood_events"
else:
    print("WARNING: could not auto-detect the concatenated event field.")
    print("Check the field list above and set 'events_field_name' manually.")
    events_field_name = None

# ------------------------------------------------------------
# 6. ADD THE REST OF THE SCHEMA
# ------------------------------------------------------------
arcpy.management.AddField(output_fc, "event_count", "SHORT")
arcpy.management.AddField(output_fc, "recurrence_class", "TEXT", field_length=15)
arcpy.management.AddField(output_fc, "flood_label", "SHORT")   # 1 = flooded, for ML training
arcpy.management.AddField(output_fc, "data_source", "TEXT", field_length=50)
arcpy.management.AddField(output_fc, "area_km2", "DOUBLE")
arcpy.management.AddField(output_fc, "flood_id", "LONG")

fields_to_update = [events_field_name or "OID@", "event_count", "recurrence_class",
                     "flood_label", "data_source", "area_m2", "area_km2", "flood_id"]

with arcpy.da.UpdateCursor(output_fc, fields_to_update) as cursor:
    for i, row in enumerate(cursor, start=1):
        events_str = row[0] if (events_field_name and row[0]) else ""
        event_list = [e.strip() for e in str(events_str).split(",") if e.strip()]
        count = len(event_list) if events_field_name else 1

        if count >= 3:
            recurrence = "High"
        elif count == 2:
            recurrence = "Moderate"
        else:
            recurrence = "Low"

        row[1] = count
        row[2] = recurrence
        row[3] = 1                                   # flood_label
        row[4] = "Sentinel-1 SAR change detection"    # data_source
        row[6] = row[5] / 1e6 if row[5] else 0        # area_km2 from area_m2
        row[7] = i                                    # flood_id
        cursor.updateRow(row)

print("Computed event_count, recurrence_class, flood_label, area_km2, flood_id")

# ------------------------------------------------------------
# 7. ARRANGE FIELD ORDER AND EXPORT A CLEAN FINAL COPY
# ------------------------------------------------------------
print("Arranging final field order...")
desired_order = ["flood_id", "flood_label", "flood_events", "event_count",
                  "recurrence_class", "area_m2", "area_km2", "data_source"]
desired_order = [f for f in desired_order if f != "flood_events" or events_field_name]

field_mappings = arcpy.FieldMappings()
field_mappings.addTable(output_fc)

for i in reversed(range(field_mappings.fieldCount)):
    fname = field_mappings.getFieldMap(i).getInputFieldName(0)
    if fname not in desired_order and fname.upper() not in ("OBJECTID", "SHAPE", "SHAPE_LENGTH", "SHAPE_AREA"):
        field_mappings.removeFieldMap(i)

output_final = "FloodInventory_Final_Clean"
arcpy.conversion.FeatureClassToFeatureClass(
    output_fc, arcpy.env.workspace, output_final, field_mapping=field_mappings
)

print(f"Final flood inventory ready: {output_final}")

# ------------------------------------------------------------
# 8. SUMMARY REPORT
# ------------------------------------------------------------
print("\n--- Flood Inventory Summary ---")
total = int(arcpy.management.GetCount(output_final)[0])
print(f"Total flood polygons: {total}")

recurrence_counts = {}
with arcpy.da.SearchCursor(output_final, ["recurrence_class"]) as cursor:
    for row in cursor:
        recurrence_counts[row[0]] = recurrence_counts.get(row[0], 0) + 1
for cls, cnt in sorted(recurrence_counts.items()):
    print(f"  {cls} recurrence: {cnt} polygons")

total_area = 0
with arcpy.da.SearchCursor(output_final, ["area_km2"]) as cursor:
    for row in cursor:
        total_area += row[0] if row[0] else 0
print(f"Total flooded area: {total_area:.2f} sq km")