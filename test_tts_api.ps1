# TTS API 테스트 스크립트

Write-Host "=== L.U.N.A. TTS API 테스트 ===" -ForegroundColor Cyan

# 1. 서버 상태 확인
Write-Host "`n[1] 서버 상태 확인..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "✓ 서버 정상 작동" -ForegroundColor Green
} catch {
    Write-Host "✗ 서버 접속 실패. 서버가 실행 중인지 확인하세요." -ForegroundColor Red
    exit
}

# 2. TTS 합성 요청
Write-Host "`n[2] TTS 합성 요청 중..." -ForegroundColor Yellow

$body = @{
    text = "こんにちは、テストです"
    model_name = "Luna"
    language = "JP"
    style = "Neutral"
    style_weight = 1.0
} | ConvertTo-Json

try {
    $headers = @{
        "Content-Type" = "application/json"
    }
    
    # 음성 파일 다운로드
    Invoke-RestMethod -Uri "http://localhost:8000/api/tts/synthesize" `
        -Method Post `
        -Headers $headers `
        -Body $body `
        -OutFile "test_output.wav"
    
    Write-Host "✓ TTS 합성 완료!" -ForegroundColor Green
    Write-Host "  파일 저장됨: test_output.wav" -ForegroundColor Cyan
    
    # 파일 자동 재생 (Windows)
    Write-Host "`n[3] 음성 파일 재생..." -ForegroundColor Yellow
    Start-Process "test_output.wav"
    
} catch {
    Write-Host "✗ TTS 합성 실패" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
