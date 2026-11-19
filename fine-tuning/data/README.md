# Combustion-agent synthetic instruction dataset (revision 2)

## Files and counts
- train.jsonl: 3,200 records
- validation.jsonl: 400 records
- test.jsonl: 400 records

Each JSONL line contains system, user, and assistant messages. The assistant message contains the complete InputParameters object, per-field provenance, explicit input anchors, and prior-status notes. Scenario group IDs and split labels exist only in record metadata and are excluded from all conversation messages. Fuel fractions are mole fractions. Fuel entries use the fields species and fraction.

## User-description variety
Prompts never mention a family number, group label, or training-set category. They use ordinary user phrasing and vary in specificity: 20% vague application-only requests, 60% partially specified requests (application/fuel plus one explicit condition or request for guidance), and 20% detailed requests (multiple explicit conditions or full ranges). Each application and fuel pattern receives examples across these levels. Split-group IDs are stored only in record metadata, to keep related scenarios on one side of the train/validation/test boundary.

## Range behavior
Pressure, temperature, and equivalence ratio are non-degenerate swept ranges in every record. If the user supplies one value, that exact number and unit are retained in explicit_anchors and the range is expanded around it; endpoints are marked explicit_and_inferred. An explicit range is copied unchanged. No unit conversion is performed.

Missing values use broad provisional application-conditioned sweeps. They are NOT validated ground truth, nominal equipment conditions, or design limits. DRGEP's autoignition sweep is a chemistry-coverage choice, distinct from actual combustor inlet or in-cylinder conditions. The prior status is explicitly marked provisional and requires domain review. Real selection depends on hardware, oxidizer, dilution, pressure basis, and study objective.

## Default broad sweeps for unspecified values
Pressure units are atm, temperature units K, and equivalence ratio dimensionless.

| Application | Pressure | Temperature | phi |
|---|---:|---:|---:|
| Stationary gas turbine | 1–25 atm | 500–2,200 K | 0.4–1.4 |
| Aircraft gas turbine | 1–40 atm | 400–2,200 K | 0.3–1.4 |
| Internal-combustion engine | 1–100 atm | 500–1,800 K | 0.3–2.0 |
| Industrial burner/furnace | 0.8–5 atm | 295–1,800 K | 0.4–1.5 |
| Boiler/power generation | 0.8–5 atm | 295–1,800 K | 0.4–1.5 |
| Rocket/aerospace | 1–100 atm | 500–3,000 K | 0.3–2.0 |
| Fundamental ignition research | 0.5–40 atm | 500–2,500 K | 0.3–2.0 |
| Hydrogen safety/release | 0.8–10 atm | 250–1,800 K | 0.2–2.0 |

These envelopes are rounded broad synthesis ranges, not measurements for each application/fuel combination. Their evidence is uneven. Published NH3/H2 shock-tube autoignition studies include about 940–2,200 K and 4–16 bar for specified compositions, dilution, and oxidizers; other work studies 1,020–1,945 K at 1.2 and 10 atm. A gas-turbine-related high-pressure shock-tube thesis reports 1,000–1,500 K near 25 bar. NASA reports facility-specific turbine inlet pressures and temperatures. A single NIST full-scale garage test is not a universal hydrogen safety envelope. These sources support the need for broad, context-specific sweeps, but do not directly validate every default row. Do not treat this dataset as an engineering recommendation.

## User-supplied example file
The attached dataset has 45 records and is used only as inspiration for species and range formatting. Its labels/ranges have no supplied methods or provenance, so they are not treated as ground truth. Its targets such as “NOx emissions” and “ignition delay” are objectives, not species names; this revision lists chemical species in target_species.

## Species and mechanisms
H2 cases retain H2/O2/H/OH/HO2/H2O/NO and target H2O/OH/HO2. NH3 cases retain NH3/NH2/N/H/NO/NO2/N2O plus relevant oxidizer/radical species and target NH3/NO/NO2/N2O. Blends combine these sets. These are chemically motivated, provisional starting lists inspired by the supplied examples; validate them against the actual mechanism and desired observables.

mechanism is null throughout because no mechanism filenames or validated compatibility map were available. The project sources directory was empty. Do not interpret this as a mechanism compatibility claim.

## Splits and validation
Eight application families each have 50 groups of 10 related examples. Per application, 40 entire groups go to train, 5 to validation, and 5 to test using deterministic SHA-256 ordering. Groups never cross splits; exact totals are 3,200/400/400. Every family appears in each split, so this is not a held-out-family evaluation.

The table gives broad family envelopes. To keep test cases distinct, the generator assigns each scenario a different pressure/temperature/phi window inside its family envelope; explicit user conditions take precedence. These per-scenario sub-windows are synthetic sweep variants, remain provisional, and should not be mistaken for measured values.

The Pydantic validator checks schema fields, positive pressure/phi, strict ranges, mole-fraction sums, and target species being retained. The generator also checks message roles, exact counts, scenario-group isolation, 4,000 unique user prompts, and 4,000 unique complete InputParameters cases. No exact prompt or input-parameter case is repeated across splits. Regenerate from project root with: python make_dataset.py

## References
- Methling et al., NH3/H2 shock-tube ignition-delay measurements, Proceedings of the Combustion Institute 41 (2025), 105835. https://doi.org/10.1016/j.proci.2025.105835
- Chen et al., Effect of hydrogen blending on high-temperature auto-ignition of ammonia at elevated pressure, Fuel (2021). https://doi.org/10.1016/j.fuel.2020.119563
- Pierro, high-pressure shock-tube tests for gas-turbine-related mixtures, University of Central Florida thesis. https://stars.library.ucf.edu/etd2020/1786/
- NASA Technical Paper 2203, facility-specific turbine test conditions. https://ntrs.nasa.gov/api/citations/19830025622/downloads/19830025622.pdf?attachment=true
- NIST, Hydrogen Release and Combustion Measurements in a Full Scale Garage. https://www.nist.gov/publications/hydrogen-release-and-combustion-measurements-full-scale-garage
- NASA, hydrogen/oxygen subscale rocket chamber test report record. https://ntrs.nasa.gov/search.jsp?R=19930001555
