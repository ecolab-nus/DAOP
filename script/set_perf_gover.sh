#!/bin/bash

for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
	echo performance | sudo tee $cpu
done

cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
