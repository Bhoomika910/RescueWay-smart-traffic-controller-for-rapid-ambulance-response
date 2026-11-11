@echo off
REM Build SUMO network from node and edge definitions
REM Run this once after installing SUMO

netconvert ^
    --node-files=intersection.nod.xml ^
    --edge-files=intersection.edg.xml ^
    --tllogic-files=intersection.tll.xml ^
    --output-file=intersection.net.xml ^
    --no-turnarounds=true

echo Network built: intersection.net.xml
pause
