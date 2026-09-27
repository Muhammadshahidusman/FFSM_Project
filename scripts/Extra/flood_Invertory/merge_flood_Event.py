"""
Merge all flood event point/polygon layers into one shapefile,
preserving an event date field for later leave-one-event-out
validation and event-count reporting.
"""

import arcpy

arcpy.env.workspace = r"C:\Users\shahid.usman\Music\Riphah_univeristy\Research Paper\Flash_Flood_Makkah\Flash_Flood_Makkah.gdb"
arcpy.env.overwriteOutput = True

flood_layers = [
    ("FloodExtent_Nov2018_Makkah_Polygons", "2018-11"),
    ("FloodExtent_Apr2021_Makkah_Polygons", "2021-04"),
    ("FloodExtent_Dec2022_Makkah_Polygons", "2022-12"),
    ("FloodExtent_Sep2024_Makkah_Polygons", "2024-09"),
    ("FloodExtent_Jan2025_Makkah_Polygons", "2025-01"),
]

# Add an "event_date" field to each layer before merging, so the
# provenance survives into the merged output
for layer, event_date in flood_layers:
    if "event_date" not in [f.name for f in arcpy.ListFields(layer)]:
        arcpy.management.AddField(layer, "event_date", "TEXT", field_length=10)
    arcpy.management.CalculateField(layer, "event_date", f"'{event_date}'", "PYTHON3")

# Merge all five into one
output_merged = "FloodPoints_AllEvents_Merged"
arcpy.management.Merge([l for l, _ in flood_layers], output_merged)

print(f"Merged {len(flood_layers)} events into: {output_merged}")

# Quick count per event, to sanity-check class balance across events
with arcpy.da.SearchCursor(output_merged, ["event_date"]) as cursor:
    from collections import Counter
    counts = Counter(row[0] for row in cursor)
for event, count in sorted(counts.items()):
    print(f"  {event}: {count} points")