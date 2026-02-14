; Inno Setup script for HG Camera Counter
; Build with: iscc packaging\windows\HGCameraCounter.iss

#define MyAppName "HG Camera Counter"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "HG"
#define MyAppExeName "HGCameraCounter.exe"
#define MyRuntimeExe "runtime_service.exe"

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

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\HGCameraCounter\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "dist\runtime_service\runtime_service.exe"; DestDir: "{app}"; DestName: "{#MyRuntimeExe}"; Flags: ignoreversion
Source: "data\config\config.yaml"; DestDir: "{app}\data\config"; Flags: onlyifdoesntexist

[Icons]
Name: "{group}\HG Camera Counter"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall HG Camera Counter"; Filename: "{uninstallexe}"
Name: "{autodesktop}\HG Camera Counter"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

