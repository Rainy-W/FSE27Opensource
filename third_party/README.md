# Third-party components

HUMER does not redistribute model weights, datasets, or the EPVD artifact.

EPVD support expects the user-provided artifact at:

```text
third_party/epvd_official/EPVD/
```

Download the official EPVD artifact from Zenodo and run
`python scripts/prepare_epvd.py --source-dir /path/to/EPVD`. The setup script
copies only the two runtime files required by the HUMER adapter. Users remain
responsible for complying with the artifact's terms.
