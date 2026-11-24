// Add to the end of your FABDEM_DEM script: exports the bare-earth DEM itself
// (2 km buffer so flow routing at the basin edge is complete) for ArcGIS processing.
Export.image.toDrive({
  image: dem.clip(roiBuffer).toFloat(),
  description: 'FABDEM_Makkah_30m_UTM37N',
  folder: 'GEE_Makkah',
  fileNamePrefix: 'FABDEM_Makkah_30m_UTM37N',
  region: roiBuffer,
  scale: 30,
  crs: 'EPSG:32637',
  maxPixels: 1e13
});
