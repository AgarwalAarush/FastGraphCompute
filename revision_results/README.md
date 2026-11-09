# Returned reviewer measurements

Import returned archives here with `revision_campaign.cli import-results`.
Each named run contains a checksummed aggregate CSV, model/diagnostic metrics,
input hashes/spectra, hardware/software provenance and coverage report.
`plot-results` writes reproducible PDFs to that run's `figures/` directory.

Read [the return instructions](../revision_campaign/RETURN_RESULTS.md).
These outputs are ignored by Git until the authors approve publication.
Do not replace the historical A100 `reproduction/results/` inputs with new
H100 measurements. Reviewed revision results can later be explicitly added
to this same performance repository as a separate dataset; manuscript figure
updates belong to `AgarwalAarush/fastgraph-paper`.
