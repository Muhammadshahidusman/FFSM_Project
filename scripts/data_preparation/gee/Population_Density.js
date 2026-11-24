// ============================================================
// Most Recent Population Density for Wadi Mehassar / Makkah ROI
// Uses GHS-POP (Global Human Settlement Layer, JRC/EC), which
// is the most current gridded population product available in
// GEE -- its 2025 epoch is the newest year offered by any open
// global population dataset here.
//
// NOTE: GHS-POP's 2025 and 2030 layers are MODEL PROJECTIONS
// (extrapolated from CIESIN GPWv4.11 census data + built-up
// area trends), not direct census-year measurements like its
// 1975-2020 layers. Worth stating that distinction in your
// methodology section.
// ============================================================

var roi = ee.FeatureCollection('projects/ee-shahidusmaninfo/assets/Makkah_ROI');
Map.centerObject(roi, 8);
Map.addLayer(roi, {}, "Study Area");

// ------------------------------------------------------------
// 1. LOAD GHS-POP 2025 (most recent epoch available)
// ------------------------------------------------------------
var pop2025 = ee.Image('JRC/GHSL/P2023A/GHS_POP/2025')
  .clip(roi)
  .rename('Population_2025');

// GHS-POP gives absolute population COUNT per 100m cell, not
// density. Convert to density (persons/km2) for use as a
// susceptibility conditioning factor, since raw counts aren't
// comparable across a mixed-terrain basin without normalizing
// for cell area.
var cellAreaKm2 = ee.Image.pixelArea().divide(1e6);
var popDensity2025 = pop2025.divide(cellAreaKm2).rename('PopDensity_2025_per_km2');

var popVis = {min: 0, max: 5000, palette: ['ffffcc', 'fed976', 'fd8d3c', 'e31a1c', '800026']};
Map.addLayer(pop2025.updateMask(pop2025.gt(0)), {min: 0, max: 50, palette: popVis.palette},
  'Population Count 2025 (per 100m cell)');
Map.addLayer(popDensity2025.updateMask(popDensity2025.gt(0)), popVis,
  'Population Density 2025 (persons/km2)');

// ------------------------------------------------------------
// 2. SUMMARY STATS -- for your methodology section
// ------------------------------------------------------------
var stats = pop2025.reduceRegion({
  reducer: ee.Reducer.sum().combine({reducer2: ee.Reducer.max(), sharedInputs: true}),
  geometry: roi.geometry(),
  scale: 100,
  maxPixels: 1e13
});
print('Total estimated resident population in ROI (2025) and max cell count:', stats);

// ------------------------------------------------------------
// 3. EXPORT TO DRIVE
// ------------------------------------------------------------
Export.image.toDrive({
  image: popDensity2025,
  description: 'PopulationDensity_GHSPOP_2025_Makkah',
  folder: 'GEE_Exports',
  fileNamePrefix: 'PopulationDensity_GHSPOP_2025_Makkah',
  region: roi.geometry(),
  scale: 100,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});