# Robot Error Code Reference

Robots report errors as codes in the form `KR-NNNN`. The first digit shows the subsystem. This reference lists the most common codes and the first remediation step for each.

## Navigation errors (KR-1xxx)

- **KR-1004 – Localization lost.** The robot cannot match its lidar scan to the site map. Usually caused by moved shelving. Remediation: re-run the site mapping job for the affected zone.
- **KR-1017 – Path blocked for over 120 seconds.** An obstacle has been detected on the planned route for more than two minutes. Remediation: ask site staff to clear the aisle; the robot will resume automatically.
- **KR-1033 – Map version mismatch.** The robot's map is older than the fleet controller's. Remediation: trigger a map sync from the fleet dashboard.

## Power errors (KR-2xxx)

- **KR-2001 – Battery below 10%.** The robot is returning to its charger. No action needed unless it repeats more than three times per shift.
- **KR-2045 – Charging contact fault.** The robot docked but did not start charging. Remediation: clean the charging contacts; if it persists, replace the dock contact plate.

## Safety errors (KR-3xxx)

- **KR-3000 – Emergency stop engaged.** Someone pressed an emergency stop button. The robot must be reset in person by trained site staff; it cannot be reset remotely.
- **KR-3012 – Safety scanner obstructed.** The front safety scanner is dirty or blocked. Remediation: clean the scanner window with the approved wipes.

## Payload errors (KR-4xxx)

- **KR-4021 – Payload overweight.** The load on the lift deck exceeds the 600 kg limit. Remediation: remove items until the load is under the limit, then retry the task.
- **KR-4050 – Lift actuator stalled.** The lift motor did not reach the target height. Remediation: take the robot out of service and open a hardware ticket.

## Communication errors (KR-5xxx)

- **KR-5002 – Fleet controller unreachable.** The robot has lost its connection to the fleet controller for more than 30 seconds. It will finish its current task and then stop safely. Remediation: check the site Wi-Fi access points.
