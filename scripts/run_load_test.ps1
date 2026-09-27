param([string]$HostUrl="http://127.0.0.1:3000",[int]$Users=10,[int]$DurationSeconds=60,[string]$Payload="tests/fixtures/pet.jpg")
$env:PET_BREED_PAYLOAD=$Payload
locust -f locustfile.py --headless -H $HostUrl -u $Users -r $Users -t "$($DurationSeconds)s" --csv reports/serving/locust --json
