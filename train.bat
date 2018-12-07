@echo off

set venvdir=C:\pyvenv\tfenv\Scripts
call %venvdir%\activate.bat
cd /D %~dp0program
python .\train.py -m ifcp_model_ver1_0 -e 5 -t 4h -st 1
robocopy %~dp0model %~dp0local_backup\m1_0\sNeNst1\model /S /MOVE
robocopy %~dp0log %~dp0local_backup\m1_0\sNeNst1\log /S /MOVE
python .\train.py -m ifcp_model_ver1_0 -e 5 -t 4h -rs "-200" -st 3
robocopy %~dp0model %~dp0local_backup\m1_0\sM200eNst3\model /S /MOVE
robocopy %~dp0log %~dp0local_backup\m1_0\sM200eNst3\log /S /MOVE
python .\train.py -m ifcp_model_ver1_1 -e 5 -t 4h -st 1
robocopy %~dp0model %~dp0local_backup\m1_1\sNeNst1\model /S /MOVE
robocopy %~dp0log %~dp0local_backup\m1_1\sNeNst1\log /S /MOVE
python .\train.py -m ifcp_model_ver1_1 -e 5 -t 4h -rs "-200" -st 3
robocopy %~dp0model %~dp0local_backup\m1_1\sM200eNst3\model /S /MOVE
robocopy %~dp0log %~dp0local_backup\m1_1\sM200eNst3\log /S /MOVE
python .\train.py -m ifcp_model_ver1_2 -e 5 -t 4h -st 1
robocopy %~dp0model %~dp0local_backup\m1_2\sNeNst1\model /S /MOVE
robocopy %~dp0log %~dp0local_backup\m1_2\sNeNst1\log /S /MOVE
python .\train.py -m ifcp_model_ver1_2 -e 5 -t 4h -st 3
robocopy %~dp0model %~dp0local_backup\m1_2\sNeNst3\model /S /MOVE
robocopy %~dp0log %~dp0local_backup\m1_2\sNeNst3\log /S /MOVE
python .\train.py -m ifcp_model_ver1_2 -e 5 -t 4h -rs "-8" -st 3
robocopy %~dp0model %~dp0local_backup\m1_2\sM8eNst3\model /S /MOVE
robocopy %~dp0log %~dp0local_backup\m1_2\sM8eNst3\log /S /MOVE
python .\train.py -m ifcp_model_ver1_2 -e 5 -t 4h -rs "-200" -st 3
robocopy %~dp0model %~dp0local_backup\m1_2\sM200eNst3\model /S /MOVE
robocopy %~dp0log %~dp0local_backup\m1_2\sM8eNst3\log /S /MOVE
deactivate
pause
exit
