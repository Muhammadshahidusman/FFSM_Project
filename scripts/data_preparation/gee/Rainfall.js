// ============================================================
// Mean Annual Total Rainfall -> Points (for IDW/Kriging in ArcGIS Pro)
// Uses the wider LargeROI (imported above) so there are enough
// sample points around/inside Wadi Mehassar for a meaningful
// interpolation -- your small basin alone only has 1-2 CHIRPS
// cells, not enough for IDW/Kriging to work with.
// ============================================================

// LargeROI is already imported above (Polygon, 4 vertices)
Map.centerObject(LargeROI, 8);
Map.addLayer(LargeROI, {}, "Large ROI (interpolation source area)");

var startYear = 1995;
var endYear   = 2024;   // 30-year climatological baseline
var years = ee.List.sequence(startYear, endYear);

// ------------------------------------------------------------
// 1. LOAD CHIRPS DAILY OVER THE FULL BASELINE PERIOD
// ------------------------------------------------------------
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(LargeROI)
  .filterDate(startYear + '-01-01', (endYear + 1) + '-01-01');

// ------------------------------------------------------------
// 2. FOR EACH YEAR, GET THAT YEAR'S TOTAL (SUM) RAINFALL
// ------------------------------------------------------------
var yearlySum = ee.ImageCollection(
  years.map(function (y) {
    y = ee.Number(y);
    var yearStart = ee.Date.fromYMD(y, 1, 1);
    var yearEnd = yearStart.advance(1, 'year');
    return chirps
      .filterDate(yearStart, yearEnd)
      .sum()
      .set('year', y)
      .set('system:time_start', yearStart.millis());
  })
);

print('Number of yearly sum images:', yearlySum.size());

// ------------------------------------------------------------
// 3. AVERAGE THE YEARLY SUMS -> MEAN ANNUAL TOTAL RAINFALL
// ------------------------------------------------------------
var meanAnnualRainfall = yearlySum.mean().clip(LargeROI)
  .rename('MeanAnnualRainfall_mm');

var rainVis = {min: 0, max: 150, palette: ['ffffcc', 'a1dab4', '41b6c4', '2c7fb8', '253494']};
Map.addLayer(meanAnnualRainfall, rainVis, 'Mean Annual Total Rainfall (1995-2024)');

// ------------------------------------------------------------
// 4. CONVERT RASTER TO POINTS
//    Each CHIRPS cell center (~5.5km) becomes a sample point
//    carrying its mean annual rainfall value -- these are your
//    interpolation "stations" for IDW/Kriging in ArcGIS Pro.
// ------------------------------------------------------------
var rainfallPoints = meanAnnualRainfall.sample({
  region: LargeROI,
  scale: 5500,          // CHIRPS native resolution -- one point per native cell
  geometries: true       // keep point geometry for export
});

print('Number of rainfall sample points:', rainfallPoints.size());
Map.addLayer(rainfallPoints, {color: 'red'}, 'Rainfall Points');

// ------------------------------------------------------------
// 5. EXPORT POINTS TO DRIVE (as Shapefile, ready for ArcGIS Pro)
// ------------------------------------------------------------
Export.table.toDrive({
  collection: rainfallPoints,
  description: 'RainfallPoints_MeanAnnual_1995_2024_Makkah',
  folder: 'rainfall_Exports',
  fileNamePrefix: 'RainfallPoints_MeanAnnual_1995_2024_Makkah',
  fileFormat: 'SHP'
});

// Also export the raster itself, in case you want to reference
// it or run RasterToPoint directly in ArcGIS Pro instead
Export.image.toDrive({
  image: meanAnnualRainfall,
  description: 'MeanAnnualRainfall_1995_2024_Makkah_Raster',
  folder: 'rainfall_Exports',
  fileNamePrefix: 'MeanAnnualRainfall_1995_2024_Makkah_Raster',
  region: LargeROI,
  scale: 5500,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});