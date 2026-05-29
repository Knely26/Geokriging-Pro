GeoKrigingPro v11.6
-------Español------------------------------------------------------------------------------------
Cambios principales:
- Kriging headless por proceso externo.
- Raster final de estimación.
- Varianza y desviación estándar de kriging generadas con soporte de bloque.
- Exporta kriging_blocks.csv con X, Y, estimación, varianza, std, número de muestras, pasada usada, fracción de dominio, capping hits y azimut LVA.
- n_samples.tif y capping_hits.tif como diagnósticos.
- pass_used.tif opcional.
- domain_fraction.tif opcional.
- Mapa variográfico circular binned con radios y ángulos visibles, más CSV de bins.
- Modelo teórico enlazado: D1 es rango mayor; ratios D2/D1 y D3/D1 se limitan internamente a <= 1.
- LVA automático 2D mejorado: ventana local, tensor de orientación, confianza LVA, fallback al azimut global si la confianza es baja, lva_azimuth.tif, lva_confidence.tif y lva_vectors para auditoría.
- Capping local con percentil global definido por usuario o MAD log.
- Radio interno automático por bloque/pasada.
- Downweight explícito con factor configurable.
- Tratamiento de negativos con percentil positivo local por bloque.

Instalación: QGIS > Complementos > Instalar desde ZIP.

- Auditoría rápida: kriging_audit.json y kriging_audit.png con estadísticas/histogramas de estimación, desviación OK, muestras usadas y confianza LVA.

---------Ingles-----------------------------------------------------------------------------------
Main changes:
- Headless kriging via external processing.

- Final estimation raster.

- Kriging variance and standard deviation generated with block support.

- Exports kriging_blocks.csv with X, Y, estimate, variance, standard deviation, number of samples, passes used, domain fraction, capping hits, and LVA azimuth.

- n_samples.tif and capping_hits.tif as diagnostics.

- pass_used.tif optional.

- domain_fraction.tif optional.

- Binned circular variographic map with visible radii and angles, plus CSV of bins.

- Linked theoretical model: D1 is the highest range; D2/D1 and D3/D1 ratios are internally limited to <= 1.
- Enhanced 2D automatic LVA: local window, orientation tensor, LVA confidence, fallback to global azimuth if confidence is low, lva_azimuth.tif, lva_confidence.tif, and lva_vectors for auditing.

- Local capping with user-defined global percentile or MAD log.

- Automatic internal radius per block/pass.

- Explicit downweighting with configurable factor.

- Handling of negative values ​​with local positive percentile per block.

Installation: QGIS > Plugins > Install from ZIP.

- Quick audit: kriging_audit.json and kriging_audit.png with estimation statistics/histograms, OK deviation, samples used, and LVA confidence.