GHIDRA_HOME=/path/to/ghidra_11.2.1_PUBLIC_20241105/ghidra_11.2.1_PUBLIC/

$GHIDRA_HOME/support/analyzeHeadless \
  /path/to/ \
  <your_project> \
  -process "<your_binary>" \
  -scriptPath /path/to/ghidra_scripts \
  -postScript ExtractFeatures

