// ============================================================
// LULC Classification using Random Forest
// Sentinel-2 Imagery, 4 Classes
// Classes: Water | Builtup | Vegetation | Barren Land
//
// Round-2 revision (Reviewer 1, minor comment b):
//   exports the accuracy assessment as CSV tables to Google Drive.
//
// RUN THIS INSIDE THE SAME SAVED GEE SCRIPT THAT HOLDS YOUR DRAWN
// GEOMETRY IMPORTS (water, builtup, vegetation, barrenLand).
// Pasted into a new script, those imports won't exist and it fails.
// If the imports were edited after the LULC map was made, the
// accuracy will not describe the published map.
//
// Outputs (Tasks tab, then Run each one):
//   1. LULC_accuracy_by_class.csv  confusion matrix rows, producer's,
//                                  user's accuracy and F1 per class
//   2. LULC_accuracy_summary.csv   overall accuracy, kappa, sample
//                                  counts, scene count and dates
//   3. LULC_test_samples.csv       every test sample with its reference
//                                  and predicted class (lets the tables
//                                  be recomputed outside GEE)
// Each table holds two validation designs:
//   pixel_split_70_30   the original design: pixels sampled from all
//                       polygons, then split 70:30 at random (seed 42).
//                       This is the classifier that made the map.
//   polygon_split_70_30 stricter check: whole reference polygons are
//                       withheld, so no test pixel shares a polygon with
//                       a training pixel. Diagnostic only; it does not
//                       change the map.
// ============================================================


// ============================================================
// STEP 1: DEFINE STUDY AREA
// ============================================================

var roi = ee.FeatureCollection('projects/ee-shahidusmaninfo/assets/Makkah_ROI');

Map.centerObject(roi, 10);
Map.addLayer(roi, {color: 'red'}, 'Study Area Boundary');


// ============================================================
// STEP 2: LOAD AND FILTER SENTINEL-2 DATA
// ============================================================

var startDate = '2025-01-01';
var endDate   = '2025-12-31';

// Cloud masking function
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask  = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
               .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask)
              .divide(10000)
              .copyProperties(image, ['system:time_start']);
}

var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
  .map(maskS2clouds);

// R2: record the scene count and date span. If the count differs from
// the run that produced the published map, the composite differs too.
var nScenes   = sentinel2.size();
var firstDate = ee.Date(sentinel2.aggregate_min('system:time_start')).format('YYYY-MM-dd');
var lastDate  = ee.Date(sentinel2.aggregate_max('system:time_start')).format('YYYY-MM-dd');
print('Total Sentinel-2 Scenes:', nScenes);
print('Scene dates from / to:', firstDate, lastDate);

var composite = sentinel2.median().clip(roi);

Map.addLayer(composite, {
  bands: ['B8', 'B4', 'B3'],
  min: 0.0,
  max: 0.3,
  gamma: 1.4
}, 'RGB Composite');


// ============================================================
// STEP 3: CALCULATE SPECTRAL INDICES
// ============================================================

var ndvi  = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi  = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
var mndwi = composite.normalizedDifference(['B3', 'B11']).rename('MNDWI');
var ndbi  = composite.normalizedDifference(['B11', 'B8']).rename('NDBI');

var bsi = composite.expression(
  '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
  {
    'BLUE':  composite.select('B2'),
    'RED':   composite.select('B4'),
    'NIR':   composite.select('B8'),
    'SWIR1': composite.select('B11')
  }
).rename('BSI');

var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];

var inputImage = composite.select(bands)
  .addBands(ndvi)
  .addBands(ndwi)
  .addBands(mndwi)
  .addBands(ndbi)
  .addBands(bsi);

print('Input bands:', inputImage.bandNames());


// ============================================================
// STEP 4: ASSIGN CLASS PROPERTIES TO DRAWN SAMPLES
//   0 Water | 1 Builtup | 2 Vegetation | 3 Barren Land
// ============================================================

var CLASS_CODES = [0, 1, 2, 3];
var CLASS_NAMES = ['Water', 'Builtup', 'Vegetation', 'BarrenLand'];

var waterFC = ee.FeatureCollection([
  ee.Feature(water, {landcover: 0, className: 'Water'})
]);
var builtupFC = ee.FeatureCollection([
  ee.Feature(builtup, {landcover: 1, className: 'Builtup'})
]);
var vegetationFC = ee.FeatureCollection([
  ee.Feature(vegetation, {landcover: 2, className: 'Vegetation'})
]);
var barrenLandFC = ee.FeatureCollection([
  ee.Feature(barrenLand, {landcover: 3, className: 'BarrenLand'})
]);


// ============================================================
// STEP 5: MERGE ALL TRAINING DATA
// ============================================================

var trainingData = waterFC
  .merge(builtupFC)
  .merge(vegetationFC)
  .merge(barrenLandFC);

print('=== TRAINING DATA SUMMARY ===');
print('Total training features:', trainingData.size());


// ============================================================
// STEP 6: SAMPLE THE INPUT IMAGE USING TRAINING POLYGONS
// ============================================================

var trainingSamples = inputImage.sampleRegions({
  collection: trainingData,
  properties: ['landcover', 'className'],
  scale: 10,
  tileScale: 16
});

print('Total samples extracted:', trainingSamples.size());
print('Samples per class:', trainingSamples.aggregate_histogram('landcover'));


// ============================================================
// STEP 7: SPLIT INTO TRAIN (70%) AND VALIDATION (30%)
// (unchanged from the original run, seed 42)
// ============================================================

trainingSamples = trainingSamples.randomColumn('random', 42);

var trainSet = trainingSamples.filter(ee.Filter.lte('random', 0.7));
var testSet  = trainingSamples.filter(ee.Filter.gt('random', 0.7));

print('Train set size:', trainSet.size());
print('Test set size:',  testSet.size());


// ============================================================
// STEP 8: TRAIN RANDOM FOREST CLASSIFIER (unchanged)
// ============================================================

var RF_PARAMS = {
  numberOfTrees:     150,
  variablesPerSplit: null,
  minLeafPopulation: 1,
  bagFraction:       0.5,
  maxNodes:          null,
  seed:              42
};

var classifier = ee.Classifier.smileRandomForest(RF_PARAMS).train({
  features:        trainSet,
  classProperty:   'landcover',
  inputProperties: inputImage.bandNames()
});


// ============================================================
// STEP 9: CLASSIFY THE IMAGE (display only)
// ============================================================

var classified = inputImage.classify(classifier);

var classVis = {
  min: 0,
  max: 3,
  palette: ['0000FF', 'FF0000', '00AA00', 'FFFF00']
};
Map.addLayer(classified.clip(roi), classVis, 'LULC Classification (4 Classes)');


// ============================================================
// STEP 10: ACCURACY TABLES (R2: rebuilt for export)
// ============================================================

// Turns a classified test collection into two FeatureCollections:
// one row per class and one summary row.
function accuracyTables(validatedFC, trainFC, splitLabel) {
  var cm  = validatedFC.errorMatrix('landcover', 'classification', CLASS_CODES);
  var arr = cm.array();                          // rows = reference, cols = predicted
  var pa  = cm.producersAccuracy();              // [n, 1]
  var ua  = cm.consumersAccuracy();              // [1, n]  (= user's accuracy)
  var f1  = cm.fscore();                         // [n]
  var rowTot = arr.reduce(ee.Reducer.sum(), [1]); // reference totals [n, 1]
  var colTot = arr.reduce(ee.Reducer.sum(), [0]); // mapped totals   [1, n]

  var perClass = ee.FeatureCollection(
    ee.List.sequence(0, CLASS_CODES.length - 1).map(function (i) {
      i = ee.Number(i).int();
      return ee.Feature(null, {
        split:            splitLabel,
        class_code:       ee.List(CLASS_CODES).get(i),
        class_name:       ee.List(CLASS_NAMES).get(i),
        ref_samples:      rowTot.get(ee.List([i, 0])),
        mapped_samples:   colTot.get(ee.List([0, i])),
        correct:          arr.get(ee.List([i, i])),
        producers_acc:    pa.get(ee.List([i, 0])),
        users_acc:        ua.get(ee.List([0, i])),
        f1:               f1.get(ee.List([i])),
        pred_Water:       arr.get(ee.List([i, 0])),
        pred_Builtup:     arr.get(ee.List([i, 1])),
        pred_Vegetation:  arr.get(ee.List([i, 2])),
        pred_BarrenLand:  arr.get(ee.List([i, 3]))
      });
    })
  );

  var summary = ee.FeatureCollection([ee.Feature(null, {
    split:            splitLabel,
    overall_accuracy: cm.accuracy(),
    kappa:            cm.kappa(),
    n_train:          trainFC.size(),
    n_test:           validatedFC.size(),
    train_by_class:   ee.String.encodeJSON(trainFC.aggregate_histogram('landcover')),
    test_by_class:    ee.String.encodeJSON(validatedFC.aggregate_histogram('landcover')),
    n_scenes:         nScenes,
    first_scene:      firstDate,
    last_scene:       lastDate,
    seed:             42
  })]);

  print('=== ACCURACY: ' + splitLabel + ' ===');
  print('Confusion matrix (rows = reference, cols = predicted):', cm);
  print('Overall accuracy:', cm.accuracy());
  print('Kappa:', cm.kappa());
  print("Producer's accuracy:", pa);
  print("User's accuracy:", ua);

  return {perClass: perClass, summary: summary};
}

// --- 10a. Original design: pixel-level 70:30 split ----------
var validated = testSet.classify(classifier);
var accPixel  = accuracyTables(validated, trainSet, 'pixel_split_70_30');


// --- 10b. Stricter check: whole polygons withheld -----------
// Each drawn import is one (multi-part) geometry, so split it into
// its parts and assign every part to train or test, class by class.
function explode(geom, code, name) {
  return ee.FeatureCollection(ee.Geometry(geom).geometries().map(function (g) {
    return ee.Feature(ee.Geometry(g), {landcover: code, className: name});
  })).randomColumn('prand', 42);
}

var polyByClass = [
  explode(water,      0, 'Water'),
  explode(builtup,    1, 'Builtup'),
  explode(vegetation, 2, 'Vegetation'),
  explode(barrenLand, 3, 'BarrenLand')
];

var trainPolys = ee.FeatureCollection([]);
var testPolys  = ee.FeatureCollection([]);
polyByClass.forEach(function (fc, k) {
  print('Reference polygons/points, ' + CLASS_NAMES[k] + ':', fc.size());
  trainPolys = trainPolys.merge(fc.filter(ee.Filter.lte('prand', 0.7)));
  testPolys  = testPolys.merge(fc.filter(ee.Filter.gt('prand', 0.7)));
});
// A class with only 1 or 2 polygons may end up with no test pixels
// (or no training pixels); its row will then show 0 or NaN.

var trainPx = inputImage.sampleRegions({
  collection: trainPolys, properties: ['landcover'], scale: 10, tileScale: 16
});
var testPx = inputImage.sampleRegions({
  collection: testPolys, properties: ['landcover'], scale: 10, tileScale: 16
});

var classifierPoly = ee.Classifier.smileRandomForest(RF_PARAMS).train({
  features:        trainPx,
  classProperty:   'landcover',
  inputProperties: inputImage.bandNames()
});
var validatedPoly = testPx.classify(classifierPoly);
var accPoly = accuracyTables(validatedPoly, trainPx, 'polygon_split_70_30');


// ============================================================
// STEP 11 (optional): AGREEMENT WITH THE PUBLISHED MAP
// Upload dataset/source/LULC_RF_4Classes_2025_30m_UTM37N.tif as an
// asset and paste its ID below. This shows whether today's rerun
// reproduces the map used in the paper (it should be close to 100%).
// ============================================================

var ORIGINAL_LULC_ASSET = '';   // e.g. 'projects/ee-shahidusmaninfo/assets/LULC_RF_4Classes_2025_30m_UTM37N'

if (ORIGINAL_LULC_ASSET !== '') {
  var orig = ee.Image(ORIGINAL_LULC_ASSET).select(0);
  orig = orig.updateMask(orig.lte(3));          // drop NoData code 15
  var rerun30 = classified
    .reduceResolution({reducer: ee.Reducer.mode(), maxPixels: 64})
    .reproject(orig.projection());
  var agreement = rerun30.eq(orig).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: roi.geometry(),
    scale: 30, maxPixels: 1e13
  });
  print('Agreement of rerun with published LULC map (share of cells):', agreement);
}


// ============================================================
// STEP 12: EXPORT ACCURACY TABLES TO GOOGLE DRIVE (R2)
// ============================================================

Export.table.toDrive({
  collection:     accPixel.perClass.merge(accPoly.perClass),
  description:    'LULC_accuracy_by_class',
  folder:         'GEE_Exports',
  fileNamePrefix: 'LULC_accuracy_by_class',
  fileFormat:     'CSV',
  selectors: ['split', 'class_code', 'class_name', 'ref_samples', 'mapped_samples',
              'correct', 'producers_acc', 'users_acc', 'f1',
              'pred_Water', 'pred_Builtup', 'pred_Vegetation', 'pred_BarrenLand']
});

Export.table.toDrive({
  collection:     accPixel.summary.merge(accPoly.summary),
  description:    'LULC_accuracy_summary',
  folder:         'GEE_Exports',
  fileNamePrefix: 'LULC_accuracy_summary',
  fileFormat:     'CSV',
  selectors: ['split', 'overall_accuracy', 'kappa', 'n_train', 'n_test',
              'train_by_class', 'test_by_class', 'n_scenes',
              'first_scene', 'last_scene', 'seed']
});

Export.table.toDrive({
  collection:     validated.map(function (f) { return f.set('split', 'pixel_split_70_30'); })
                    .merge(validatedPoly.map(function (f) { return f.set('split', 'polygon_split_70_30'); })),
  description:    'LULC_test_samples',
  folder:         'GEE_Exports',
  fileNamePrefix: 'LULC_test_samples',
  fileFormat:     'CSV',
  selectors:      ['split', 'landcover', 'classification']
});


// ============================================================
// STEP 13: MAP EXPORTS (switched off)
// The published map already exists. Exporting again would only risk
// mixing a rerun with the original, so these stay commented out.
// ============================================================

// Export.image.toDrive({
//   image:          classified.clip(roi).toByte(),
//   description:    'LULC_RF_4Classes_2025_rerun',
//   folder:         'GEE_Exports',
//   fileNamePrefix: 'LULC_RF_4Classes_2025_rerun',
//   region:         roi,
//   scale:          10,
//   maxPixels:      1e13,
//   crs:            'EPSG:4326'
// });

print('=== SCRIPT COMPLETE: go to the Tasks tab and run the 3 table exports ===');
