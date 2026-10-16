param($docx,$pdf)
$w = New-Object -ComObject Word.Application
$w.Visible = $false; $w.DisplayAlerts = 0
try {
  $d = $w.Documents.Open($docx, $false, $true)   # solo lectura
  $d.ExportAsFixedFormat($pdf, 17)
  $d.Close($false)
} finally { $w.Quit() }
