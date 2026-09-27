// ============================================================
// Sentinel-2 NDVI (2025) for Wadi Mehassar / Makkah ROI
// Cloud-masked median composite -> NDVI -> export to Drive
// ============================================================

// Define a study area of interest
var roi = ee.FeatureCollection('projects/ee-shahidusmaninfo/assets/Makkah_ROI');
Map.centerObject(roi, 8);
Map.addLayer(roi, {}, "Study Area");

// ------------------------------------------------------------
// 1. CLOUD MASK FUNCTION (Sentinel-2 SR uses the QA60 band /
//    Cloud Score+ if available; here we use the standard QA60
//    bitmask approach, which works across the full archive)
// ------------------------------------------------------------
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000)
      .copyProperties(image, ['system:time_start']);
}

// ------------------------------------------------------------
// 2. LOAD SENTINEL-2 SURFACE REFLECTANCE, FILTERED TO 2025
// ------------------------------------------------------------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2025-01-01', '2025-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);

print('Number of Sentinel-2 images (2025):', s2.size());

// ------------------------------------------------------------
// 3. BUILD A CLOUD-FREE MEDIAN COMPOSITE, CLIPPED TO ROI
// ------------------------------------------------------------
var composite = s2.median().clip(roi);

// ------------------------------------------------------------
// 4. COMPUTE NDVI  =  (NIR - RED) / (NIR + RED)
//    Sentinel-2: B8 = NIR, B4 = Red
// ------------------------------------------------------------
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');

// ------------------------------------------------------------
// 5. VISUALIZE
// ------------------------------------------------------------
var ndviVis = {
  min: -0.2,
  max: 0.8,
  palette: ['blue', 'white', 'green']
};
Map.addLayer(ndvi, ndviVis, 'NDVI 2025');

// True-color composite for reference
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3},
  'True Color 2025');

// ------------------------------------------------------------
// 6. EXPORT NDVI RASTER TO GOOGLE DRIVE
// ------------------------------------------------------------
Export.image.toDrive({
  image: ndvi,
  description: 'NDVI_Makkah_2025',
  folder: 'GEE_Exports',
  fileNamePrefix: 'NDVI_Makkah_2025',
  region: roi.geometry(),
  scale: 10,                 // Sentinel-2 native resolution for B4/B8
  crs: 'EPSG:4326',
  maxPixels: 1e13
});