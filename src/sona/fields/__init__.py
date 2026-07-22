"""sona.fields — magnetic-field providers.

One generic CSV-based provider (csv_field.CSVFieldProvider) handles every
source: a synthetic test field, an OPERA2D export of the simplified
two-solenoid geometry, or later the full OPPIS geometry. The propagator
only ever talks to the FieldProvider interface defined in base.py.
"""