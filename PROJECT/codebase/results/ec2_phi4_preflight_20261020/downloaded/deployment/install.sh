#!/bin/bash
set -euo pipefail
cd /home/ubuntu/ethics-ec2
sudo install -d -m 755 /usr/local/lib/ethics-ec2
sudo install -m 644 PROJECT/codebase/src/analysis/ec2_job_lifecycle.py /usr/local/lib/ethics-ec2/lifecycle.py
for case in lifecycle-success lifecycle-failure lifecycle-timeout phi4-smoke backup-deadline; do
  install -d -m 755 "runs/$case"
done
sudo install -m 644 deployment/*.service deployment/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/ethics-*.service /etc/systemd/system/ethics-*.timer
echo EC2_UNITS_INSTALLED_NOT_STARTED
