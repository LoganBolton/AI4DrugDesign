# Logging Guide

The integrated drug discovery pipeline now includes comprehensive logging for debugging.

## Log File Location

All logs are written to: **`drug_discovery_pipeline.log`** (in the project root directory)

Logs are also printed to the console (stdout) when running the app.

## Log Levels

- **INFO**: Normal pipeline progress (step starts/completions, counts, key milestones)
- **DEBUG**: Detailed information (API responses, batch processing details)
- **WARNING**: Unexpected but non-critical issues (missing data, fallback behaviors)
- **ERROR**: Errors that should be investigated (API failures, exceptions)

## What Gets Logged

### Step 1: Protein Analysis
- ✅ PDB ID being analyzed
- ✅ UniProt mapping results
- ✅ Number of organisms, ligands, binding sites found
- ✅ AI analysis requests and completions
- ✅ Any errors fetching protein data

### Step 2: Compound Discovery
- ✅ ChEMBL target lookup (UniProt → ChEMBL target ID)
- ✅ Activity fetching progress (batch offsets)
- ✅ Number of compounds fetched from ChEMBL
- ✅ PDB ligand fetching
- ✅ Total compounds discovered
- ✅ API errors and timeouts

### Step 3: Rule of 5 Filter
- ✅ Number of input compounds
- ✅ Number passing Rule of 5
- ✅ Filter completion

### Step 4: Binding Activity Ranking
- ✅ Number of compounds with/without binding data
- ✅ Ranking completion

### Step 5: ADME Filter
- ✅ Number of input compounds
- ✅ Number passing ADME criteria
- ✅ Filter completion

### Step 6: Compound Selection & AI
- ✅ Which compound was selected
- ✅ AI explanation requests
- ✅ AI completion or errors

## Example Log Output

```
2026-03-17 10:30:15 - __main__ - INFO - === STEP 1: Analyzing protein 6LU7 ===
2026-03-17 10:30:15 - __main__ - INFO - Fetching protein info for PDB ID: 6LU7
2026-03-17 10:30:16 - __main__ - DEBUG - Retrieved entry data for 6LU7
2026-03-17 10:30:16 - __main__ - DEBUG - Found 1 organism(s) for 6LU7
2026-03-17 10:30:17 - __main__ - INFO - Found 0 ligand(s) in 6LU7
2026-03-17 10:30:18 - __main__ - INFO - Found 1 binding site(s) in 6LU7
2026-03-17 10:30:18 - __main__ - INFO - Successfully fetched complete protein info for 6LU7
2026-03-17 10:30:18 - __main__ - INFO - UniProt ID for 6LU7: P0DTD1
2026-03-17 10:30:18 - __main__ - INFO - Requesting AI analysis for 6LU7
2026-03-17 10:30:22 - __main__ - INFO - AI analysis completed for 6LU7
2026-03-17 10:30:22 - __main__ - INFO - === STEP 1 COMPLETE: 6LU7 analyzed successfully ===

2026-03-17 10:30:45 - __main__ - INFO - === STEP 2: Fetching compounds ===
2026-03-17 10:30:45 - __main__ - INFO - Fetching compounds for 6LU7 (UniProt: P0DTD1)
2026-03-17 10:30:45 - __main__ - INFO - Looking up ChEMBL target for UniProt P0DTD1
2026-03-17 10:31:15 - __main__ - INFO - Found ChEMBL target: CHEMBL4523582
2026-03-17 10:31:15 - __main__ - INFO - Fetching activities for ChEMBL target CHEMBL4523582
2026-03-17 10:31:15 - __main__ - DEBUG - Fetching activities batch at offset 0
2026-03-17 10:31:45 - __main__ - DEBUG - Fetching activities batch at offset 500
2026-03-17 10:32:10 - __main__ - DEBUG - Fetching activities batch at offset 1000
2026-03-17 10:32:35 - __main__ - DEBUG - Fetching activities batch at offset 1500
2026-03-17 10:32:55 - __main__ - DEBUG - Reached end of activities (got 428 in last batch)
2026-03-17 10:32:55 - __main__ - INFO - Fetched 1928 compounds from ChEMBL activities
2026-03-17 10:32:55 - __main__ - INFO - Fetching co-crystallized ligands from PDB 6LU7
2026-03-17 10:32:56 - __main__ - INFO - === STEP 2 COMPLETE: 1928 compounds discovered ===

2026-03-17 10:33:10 - __main__ - INFO - === STEP 3: Applying Rule of 5 filter ===
2026-03-17 10:33:10 - __main__ - INFO - Filtering 1928 compounds by Rule of 5
2026-03-17 10:33:12 - __main__ - INFO - === STEP 3 COMPLETE: 1547/1928 compounds passed Rule of 5 ===

2026-03-17 10:33:20 - __main__ - INFO - === STEP 4: Ranking by binding activity ===
2026-03-17 10:33:20 - __main__ - INFO - Ranking 1547 compounds: 1523 with binding data, 24 without
2026-03-17 10:33:21 - __main__ - INFO - === STEP 4 COMPLETE: 1547 compounds ranked ===

2026-03-17 10:33:30 - __main__ - INFO - === STEP 5: Applying ADME filter ===
2026-03-17 10:33:30 - __main__ - INFO - Filtering 1547 compounds by ADME criteria
2026-03-17 10:33:32 - __main__ - INFO - === STEP 5 COMPLETE: 142/1547 compounds passed ADME filter ===

2026-03-17 10:34:05 - __main__ - INFO - Selected compound: Nirmatrelvir (ChEMBL: CHEMBL4523582)
2026-03-17 10:34:12 - __main__ - INFO - === Generating AI compound explanation ===
2026-03-17 10:34:12 - __main__ - INFO - Generating AI explanation for Nirmatrelvir targeting 6LU7
2026-03-17 10:34:18 - __main__ - INFO - AI compound explanation completed successfully
```

## Viewing Logs

### View all logs:
```bash
cat drug_discovery_pipeline.log
```

### View in real-time (tail):
```bash
tail -f drug_discovery_pipeline.log
```

### View only errors:
```bash
grep ERROR drug_discovery_pipeline.log
```

### View only warnings and errors:
```bash
grep -E "WARNING|ERROR" drug_discovery_pipeline.log
```

### View step completions:
```bash
grep "COMPLETE" drug_discovery_pipeline.log
```

## Debugging Tips

1. **Pipeline stuck?** Check the log to see which step it's on
2. **No compounds found?** Look for ChEMBL target ID in logs - if "Not found", try a different protein
3. **API timeouts?** Look for ERROR entries with timeout messages
4. **Unexpected results?** Check the compound counts at each step to see where filtering is happening

## Configuration

To change log level to DEBUG (more verbose), edit `tabs/integrated_pipeline.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change INFO to DEBUG
    ...
)
```

To disable console output (only write to file):

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_discovery_pipeline.log'),
        # Remove: logging.StreamHandler()
    ]
)
```
