import gunpowder as gp

raw = gp.ArrayKey("RAW")
request = gp.BatchRequest()
request.add(raw, gp.Coordinate(5, 100, 100))

# Define the Zarr source
zarr_source = gp.ZarrSource(
    'path/to/your/data.zarr',
    datasets={
        raw: '3d/raw'
    },
    array_specs={
        raw: gp.ArraySpec(interpolatable=True)
    }
)

# Select a subset of the data
subset = gp.Slice(raw, (slice(0, 5), slice(100, 200), slice(100, 200)))

# Permute the axes to treat the first dimension as non-spatial
permute = gp.Permute(raw, (1, 2, 0))

# Add the Zarr source and transformations to the pipeline
pipeline = (
    zarr_source +
    subset +
    permute +
    gp.RandomProvider() +
    gp.SimpleAugment() +
    gp.IntensityAugment(raw, 0.7, 1.3, -0.2, 0.2) +
    gp.ElasticAugment(
        control_point_spacing=(32,)*2,
        jitter_sigma=(2.,)*2,
        rotation_interval=(0, math.pi/2),
        scale_interval=(0.8, 1.2),
        spatial_dims=2
    ) +
    gp.NoiseAugment(raw, var=0.01) +
    CreateMask()
)

with gp.build(pipeline):
    batch = pipeline.request_batch(request)
