// ============================================================
// Sentinel-1 SAR Flood Extent Detection -- Advanced Version
// Wadi Mehassar / Makkah ROI
//
// TARGET EVENT: 22–23 December 2022 Makkah Flash Flood
// (Overnight torrential rain on 22 Dec led to flooding in Makkah’s
// Al-Otaibi neighborhood on 23 Dec — cars swept away, Grand Mosque
// courtyard flooded. Distinct from the 24 Nov 2022 event centered
// on Jeddah.)
//
// NOTE: Permanent water is shown as a reference layer only — it is
// NOT used to mask the detected flood extent, per your instruction.
// ============================================================

var roi = ee.FeatureCollection('projects/ee-shahidusmaninfo/assets/Makkah_ROI');
Map.centerObject(roi, 10);
Map.addLayer(roi, {color: 'black'}, "Study Area Boundary", true, 0.6);

// --- TARGET: 22–23 December 2022 Makkah Flash Flood ---
var floodDate   = '2022-12-23';   // flooding peaked the morning of Dec 23
var beforeStart = '2022-12-01';   // dry baseline window, well before the event
var beforeEnd   = '2022-12-15';
var afterStart  = '2022-12-23';   // start on the flood day itself
var afterEnd    = '2023-01-04';   // wide window to catch enough S1 passes

// ------------------------------------------------------------
// 1. SENTINEL-1 GRD (VH POLARIZATION) -- BEFORE / AFTER
// ------------------------------------------------------------
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select('VH');

var beforeCollection = s1.filterDate(beforeStart, beforeEnd);
var afterCollection  = s1.filterDate(afterStart, afterEnd);

print('S1 before-flood images:', beforeCollection.size());
print('S1 after-flood images:', afterCollection.size());
print('S1 after-flood image dates:', afterCollection.aggregate_array('system:time_start')
  .map(function(t){ return ee.Date(t).format('YYYY-MM-dd'); }));

var beforeComposite = beforeCollection.mean().clip(roi);
var afterComposite  = afterCollection.mean().clip(roi);

var smoothingRadius = 50;
var beforeFiltered = beforeComposite.focal_mean(smoothingRadius, 'circle', 'meters');
var afterFiltered  = afterComposite.focal_mean(smoothingRadius, 'circle', 'meters');

var sarVis = {min: -25, max: 0};
Map.addLayer(beforeFiltered, sarVis, 'SAR VH - Before Flood', false);
Map.addLayer(afterFiltered, sarVis, 'SAR VH - After Flood', false);

// ------------------------------------------------------------
// 2. FLOOD DETECTION: RATIO OF AFTER/BEFORE
// ------------------------------------------------------------
var ratio = afterFiltered.divide(beforeFiltered);
Map.addLayer(ratio, {min: 0, max: 2, palette: ['red', 'white', 'blue']},
  'SAR Change Ratio (After/Before)', false);

var floodThreshold = 1.25;
var flooded = ratio.gt(floodThreshold).rename('flooded').selfMask();

// Permanent water shown as REFERENCE ONLY — not used to mask the
// flood layer below, per your instruction
var permanentWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('seasonality').gte(10);
Map.addLayer(permanentWater.selfMask(), {palette: ['00FFFF']}, 'Permanent Water (reference only)', false);

Map.addLayer(flooded, {palette: ['0000FF']}, 'Detected Flood Extent (SAR)', true);

// ------------------------------------------------------------
// 3. SENTINEL-2 TRUE COLOR -- BEFORE / AFTER (visual confirmation)
// ------------------------------------------------------------
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start']);
}

var s2Before = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(beforeStart, beforeEnd)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2clouds)
  .median()
  .clip(roi);

var s2After = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(afterStart, afterEnd)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2clouds)
  .median()
  .clip(roi);

var rgbVis = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3};
Map.addLayer(s2Before, rgbVis, 'Sentinel-2 True Color - Before', false);
Map.addLayer(s2After, rgbVis, 'Sentinel-2 True Color - After', false);

// ------------------------------------------------------------
// 4. VECTORIZE FLOOD EXTENT (for training points later)
// ------------------------------------------------------------
var floodVectors = flooded.reduceToVectors({
  geometry: roi.geometry(),
  scale: 10,
  geometryType: 'polygon',
  eightConnected: false,
  labelProperty: 'flooded',
  maxPixels: 1e13
});

print('Number of flood polygons:', floodVectors.size());

// --- RECOMMENDED: Raster-based total flooded area (sq km) ---
var floodedArea = flooded.multiply(ee.Image.pixelArea())
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: roi.geometry(),
    scale: 10,
    maxPixels: 1e13,
    bestEffort: true
  })
  .get('flooded');

print('Total flooded area (sq km):', ee.Number(floodedArea).divide(1e6));

Map.addLayer(floodVectors.style({color: 'red', fillColor: '00000000', width: 2}),
  {}, 'Flood Polygon Outline', true);

// ------------------------------------------------------------
// 5. EXPORT
// ------------------------------------------------------------
Export.image.toDrive({
  image: flooded,
  description: 'FloodExtent_Dec2022_Makkah_Raster',
  folder: 'GEE_Exports',
  fileNamePrefix: 'FloodExtent_Dec2022_Makkah_Raster',
  region: roi.geometry(),
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.table.toDrive({
  collection: floodVectors,
  description: 'FloodExtent_Dec2022_Makkah_Polygons',
  folder: 'GEE_Exports',
  fileNamePrefix: 'FloodExtent_Dec2022_Makkah_Polygons',
  fileFormat: 'SHP'
});