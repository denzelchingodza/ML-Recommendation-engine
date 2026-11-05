# Data source

**Crunchbase 2013 Snapshot © 2013**, licensed under
[Creative Commons Attribution (CC-BY)](https://creativecommons.org/licenses/by/4.0/).
Data from [Crunchbase](https://www.crunchbase.com).

These files are the October 2013 export, copied unchanged from
[datahoarder/crunchbase-october-2013](https://github.com/datahoarder/crunchbase-october-2013)
at commit `9303ebdc38fd6dbaa597e12a81edeec0246a61e2`. The export predates
Crunchbase's December 2013 switch to a non-commercial licence, so CC-BY applies.

## Files and fingerprints (SHA-256)

- `crunchbase-investments.csv` (10,339,663 bytes):
  `8f13f28d35915633c0a814dd22d49540ee115717d10d387ebc3a49183a35b65f`
- `crunchbase-companies.csv` (2,544,744 bytes):
  `83217e8165db15124aec2799f5f28bda910ab09cc1ed9553b066eb97f60f195d`

The loader checks these fingerprints every time. If a single byte changes,
loading stops with a clear error.

## Known quirks

- **Encoding:** the files are Mac Roman, not UTF-8. Read as Mac Roman,
  "MiaSolé" appears correctly.
- **Garbled names in the source:** a few names were already broken before
  export (for example `agÃ¡mi Systems`). They are left as they are, because
  guessing a fix could change real names.
