@echo off
for /l %%i in (1,1,6) do (
  start /B det experiment create const.yaml . -f
)