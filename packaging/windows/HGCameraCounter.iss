; Inno Setup script for HG Camera Counter (single-exe build)
; Build with: iscc packaging\windows\HGCameraCounter.iss
; Prereq: run packaging\windows\build_exe.bat first (produces dist\HGCameraCounter\).

#define MyAppName "HG Camera Counter"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "HG"
#define MyAppExeName "HGCameraCounter.exe"

[Setup]
AppId={{A84E0D4A-D75D-4BD8-B501-8CB7B433DAAB}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\HGCameraCounter
DefaultGroupName=HG Camera Counter
OutputDir=dist\installer
OutputBaseFilename=HGCameraCounter_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=assets\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; PyInstaller onedir output (HGCameraCounter.exe + _internal with torch/ultralytics/PySide6)
Source: "dist\HGCameraCounter\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
; Runtime assets that live next to the exe (the frozen app uses the exe dir as project root)
Source: "assets\*"; DestDir: "{app}\assets"; Flags: recursesubdirs ignoreversion skipifsourcedoesntexist
Source: "models\*.pt"; DestDir: "{app}\models"; Flags: ignoreversion skipifsourcedoesntexist
Source: "tools\ffmpeg\*"; DestDir: "{app}\tools\ffmpeg"; Excludes: "*.zip,doc\*"; Flags: recursesubdirs ignoreversion skipifsourcedoesntexist
Source: "data\zones\*"; DestDir: "{app}\data\zones"; Flags: recursesubdirs ignoreversion skipifsourcedoesntexist
; Ship the TEMPLATE only — the real config.yaml is machine-bound (DPAPI-encrypted secrets)
; and is provisioned per device (Setup Wizard), never copied between machines.
Source: "data\config\config.template.yaml"; DestDir: "{app}\data\config"; Flags: skipifsourcedoesntexist onlyifdoesntexist

[Icons]
Name: "{group}\HG Camera Counter"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\app_icon.ico"
Name: "{group}\Uninstall HG Camera Counter"; Filename: "{uninstallexe}"
Name: "{autodesktop}\HG Camera Counter"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\app_icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
