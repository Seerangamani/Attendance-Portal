package com.Attendance.BackEnd_Attendance.Service;

import com.Attendance.BackEnd_Attendance.Model.*;
import com.Attendance.BackEnd_Attendance.Repository.*;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestBody;

import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class AttendanceService {

    @Autowired
    private AttendanceRepository attendanceRepository;

    @Autowired
    private EmployeeRepository employeeRepository;

    @Autowired
    private EmployeeAttendanceRepository repository;

    @Autowired
    private MonthlyPermissionTrackerRepository permissionTrackerRepository;
//
//    @Autowired
//    private PermissionHalfdayRepository permissionHalfdayRepository;

//    private static final LocalTime GRACE_PERIOD_END = LocalTime.of(9, 30); // 9:30 AM
//    private static final LocalTime OFFICIAL_START_TIME = LocalTime.of(9, 0); // 9:00 AM
//    private static final LocalTime SPECIAL_EMPLOYEE_START = LocalTime.of(9, 40); // 9:40 AM for mmw027
//
//
//    // Time formatters
//    private static final DateTimeFormatter TIME_FORMAT_12_HOUR = DateTimeFormatter.ofPattern("h:mm a");
//    private static final DateTimeFormatter TIME_FORMAT_24_HOUR = DateTimeFormatter.ofPattern("HH:mm");


    // Helper method to convert "On Leave" status to "Absent" for display
    private String getDisplayStatus(String originalStatus) {
        if ("On Leave".equals(originalStatus)) {
            return "Absent";
        }
        return originalStatus;
    }
    /**
     * ✅ UPDATED: Enhanced checkIn method with gender-based timing validation
     */
    public Attendance checkIn(Map<String, String> body) {
        String employeeId = body.get("employeeId");
        String time = body.get("time");

        System.out.println("=== CHECK-IN DEBUG ===");
        System.out.println("Employee ID: " + employeeId);
        System.out.println("Time: " + time);

        if (employeeId == null || time == null) {
            throw new IllegalArgumentException("Missing employeeId or time");
        }

        // Validate employee exists
        Employee employee = employeeRepository.findById(employeeId)
                .orElseThrow(() -> new IllegalArgumentException("Employee not found"));

        LocalDate today = LocalDate.now();
        LocalTime checkinTime = LocalTime.now().withNano(0);
        System.out.println("Today's date: " + today);
        System.out.println("Current checkin time: " + checkinTime);

        // ✅ CRITICAL: Check if attendance already exists for today
        List<Attendance> existingRecords = attendanceRepository.findByEmployeeIdAndDate(employeeId, today);
        System.out.println("Existing records found: " + existingRecords.size());

        if (!existingRecords.isEmpty()) {
            Attendance existingAttendance = existingRecords.get(0);
            System.out.println("Existing attendance status: " + existingAttendance.getStatus());
            System.out.println("Existing checkin time: " + existingAttendance.getCheckinTime());

            // ✅ NEW LOGIC: Check if employee has permission or half-day allocation
            String currentStatus = existingAttendance.getStatus();

            // ✅ UPDATED: Check if employee is already marked as Absent - don't change status
            if ("Absent".equalsIgnoreCase(currentStatus)) {
                System.out.println("Employee is already marked as Absent - Status remains: Absent");
                existingAttendance.setCheckinTime(checkinTime);
                existingAttendance.setStatus("Absent"); // Keep status as Absent
                return attendanceRepository.save(existingAttendance);
            }

            // Handle Permission allocation
            if ("Permission".equalsIgnoreCase(currentStatus) &&
                    existingAttendance.getPermissionFrom() != null &&
                    existingAttendance.getPermissionTo() != null) {

                LocalTime permissionFrom = existingAttendance.getPermissionFrom();
                LocalTime permissionTo = existingAttendance.getPermissionTo();

                System.out.println("Permission allocated from: " + permissionFrom + " to: " + permissionTo);
                System.out.println("Employee checkin time: " + checkinTime);

                // Check if employee comes within allocated permission time
                if (checkinTime.isAfter(permissionFrom) && checkinTime.isBefore(permissionTo)) {
                    // Employee came within permission time - keep status as Permission
                    existingAttendance.setCheckinTime(checkinTime);
                    existingAttendance.setStatus("Permission"); // Keep same status
                    System.out.println("Employee came within permission time - Status remains: Permission");
                } else {
                    // Employee came outside permission time - mark as Late
                    existingAttendance.setCheckinTime(checkinTime);
                    existingAttendance.setStatus("Late");
                    System.out.println("Employee came outside permission time - Status changed to: Late");
                }

                return attendanceRepository.save(existingAttendance);
            }

            // Handle Half Day allocation
            else if ("Half Day Leave".equalsIgnoreCase(currentStatus) &&
                    existingAttendance.getHalfdayFrom() != null &&
                    existingAttendance.getHalfdayTo() != null) {

                LocalTime halfdayFrom = existingAttendance.getHalfdayFrom();
                LocalTime halfdayTo = existingAttendance.getHalfdayTo();

                System.out.println("Half day allocated from: " + halfdayFrom + " to: " + halfdayTo);
                System.out.println("Employee checkin time: " + checkinTime);

                // Check if employee comes within allocated half-day time
                if (checkinTime.isAfter(halfdayFrom) && checkinTime.isBefore(halfdayTo)) {
                    // Employee came within half-day time - keep status as Half Day Leave
                    existingAttendance.setCheckinTime(checkinTime);
                    existingAttendance.setStatus("Half Day Leave"); // Keep same status
                    System.out.println("Employee came within half-day time - Status remains: Half Day Leave");
                } else {
                    // Employee came outside half-day time - mark as Absent
                    existingAttendance.setCheckinTime(checkinTime);
                    existingAttendance.setStatus("Absent");
                    System.out.println("Employee came outside half-day time - Status changed to: Absent");
                }

                return attendanceRepository.save(existingAttendance);
            }

            // For other existing statuses (already checked in normally)
            else {
                String displayStatus = getDisplayStatus(existingAttendance.getStatus());
                throw new IllegalArgumentException(
                        String.format("Attendance already registered for today. Current status: %s, Check-in time: %s",
                                displayStatus,
                                existingAttendance.getCheckinTime() != null ? existingAttendance.getCheckinTime().toString() : "Not available")
                );
            }
        }

        // ✅ Create new attendance record with gender-based timing (normal flow)
        String status = determineAttendanceStatusWithGender(employeeId, checkinTime);

        System.out.println("Creating new attendance record with checkin time: " + checkinTime);
        System.out.println("Determined status: " + status);

        Attendance attendance = new Attendance();
        attendance.setEmployeeId(employeeId);
        attendance.setEmployeeName(employee.getUsername());
        attendance.setDepartment(employee.getDepartment());
        attendance.setDesignation(employee.getDesignation());
        attendance.setDate(today);
        attendance.setCheckinTime(checkinTime);
        attendance.setStatus(status);

        Attendance saved = attendanceRepository.save(attendance);
        System.out.println("Saved attendance record with ID: " + saved.getId());

        return saved;
    }

    /**
     * ✅ Enhanced method: Check attendance status - CONVERTS "On Leave" to "Absent" for display
     */
    public Map<String, Object> checkAttendanceStatus(String employeeId, LocalDate date) {
        Map<String, Object> response = new HashMap<>();

        try {
            System.out.println("=== CHECKING ATTENDANCE STATUS ===");
            System.out.println("Employee ID: " + employeeId);
            System.out.println("Date: " + date);

            List<Attendance> attendanceRecords = attendanceRepository.findByEmployeeIdAndDate(employeeId, date);
            System.out.println("Records found: " + attendanceRecords.size());

            if (attendanceRecords.isEmpty()) {
                response.put("isMarked", false);
                response.put("message", "No attendance record found for " + date.format(DateTimeFormatter.ISO_LOCAL_DATE));
                response.put("employeeId", employeeId);
                response.put("date", date.format(DateTimeFormatter.ISO_LOCAL_DATE));
                return response;
            }

            // Attendance record exists
            Attendance attendance = attendanceRecords.get(0);
            System.out.println("Found attendance record: ID=" + attendance.getId() + ", Status=" + attendance.getStatus());

            response.put("isMarked", true);
            response.put("message", "Attendance already marked for " + date.format(DateTimeFormatter.ISO_LOCAL_DATE));
            response.put("employeeId", employeeId);
            response.put("employeeName", attendance.getEmployeeName());
            response.put("department", attendance.getDepartment());
            response.put("designation", attendance.getDesignation());
            response.put("date", date.format(DateTimeFormatter.ISO_LOCAL_DATE));
            response.put("currentStatus", attendance.getStatus()); // ✅ No conversion needed since we use "Absent" directly
            response.put("checkinTime", attendance.getCheckinTime() != null ?
                    attendance.getCheckinTime().format(DateTimeFormatter.ofPattern("HH:mm:ss")) : null);
            response.put("checkoutTime", attendance.getCheckoutTime() != null ?
                    attendance.getCheckoutTime().format(DateTimeFormatter.ofPattern("HH:mm:ss")) : null);

            return response;

        } catch (Exception e) {
            System.err.println("Error in checkAttendanceStatus: " + e.getMessage());
            e.printStackTrace();

            response.put("isMarked", false);
            response.put("message", "Error checking attendance status: " + e.getMessage());
            response.put("error", true);
            return response;
        }
    }

    /**
     * ✅ New method: Check if attendance exists using boolean return
     */
    public boolean isAttendanceAlreadyMarked(String employeeId, LocalDate date) {
        try {
            boolean exists = attendanceRepository.existsByEmployeeIdAndDate(employeeId, date);
            System.out.println("Attendance exists for " + employeeId + " on " + date + ": " + exists);
            return exists;
        } catch (Exception e) {
            System.err.println("Error checking if attendance exists: " + e.getMessage());
            return false;
        }
    }

    /**
     * ✅ Enhanced method: Get attendance details for validation
     */
    public Optional<Attendance> getAttendanceForDate(String employeeId, LocalDate date) {
        try {
            List<Attendance> records = attendanceRepository.findByEmployeeIdAndDate(employeeId, date);
            System.out.println("Getting attendance for " + employeeId + " on " + date + ": " + records.size() + " records found");

            if (!records.isEmpty()) {
                Attendance attendance = records.get(0);
                System.out.println("Found attendance: ID=" + attendance.getId() +
                        ", CheckinTime=" + attendance.getCheckinTime() +
                        ", Status=" + attendance.getStatus());
                return Optional.of(attendance);
            }
            return Optional.empty();
        } catch (Exception e) {
            System.err.println("Error getting attendance for date: " + e.getMessage());
            return Optional.empty();
        }
    }

    /**
     * ✅ FIXED: Private method: Determine attendance status based on check-in time
     * This method only determines status for employees who are actually checking in
     */
    private String determineAttendanceStatus(LocalTime checkinTime) {
        LocalTime presentThreshold = LocalTime.of(9, 30);   // 9:30 AM
        LocalTime lateThreshold = LocalTime.of(11, 0);      // 11:00 AM

        if (checkinTime.isBefore(presentThreshold)) {
            return "Present";
        } else if (checkinTime.isBefore(lateThreshold)) {
            return "Late";
        } else {
            return "On Duty"; // Very late arrivals (but still working)
        }
    }

    /**
     * ✅ UPDATED: Enhanced recordCheckout method - Uses "Absent" instead of "On Leave"
     */
    public String recordCheckout(AttendanceRequest request) {
        String employeeId = request.getEmployeeId();
        String timeStr = request.getTime();

        System.out.println("=== CHECKOUT DEBUG ===");
        System.out.println("Employee ID: " + employeeId);
        System.out.println("Time: " + timeStr);

        if (employeeId == null || timeStr == null) {
            System.out.println("Missing employeeId or time");
            return "Missing employeeId or time";
        }

        LocalDate today = LocalDate.now();
        System.out.println("Today's date: " + today);

        // ✅ Get employee work timing based on gender
        Map<String, LocalTime> workTiming = getEmployeeWorkTiming(employeeId);
        LocalTime expectedEndTime = workTiming.get("endTime");
        System.out.println("Expected end time for employee: " + expectedEndTime);

        // ✅ Try multiple approaches to find attendance record
        List<Attendance> attendanceList = null;

        try {
            // Primary method
            attendanceList = attendanceRepository.findByEmployeeIdAndDate(employeeId, today);
            System.out.println("Primary search - Records found: " + attendanceList.size());

            // If no records found, try alternative methods
            if (attendanceList.isEmpty()) {
                System.out.println("No records found with primary method, trying alternatives...");

                // Try with Optional method
                Optional<Attendance> optionalAttendance = attendanceRepository.findAttendanceByEmployeeIdAndDate(employeeId, today);
                if (optionalAttendance.isPresent()) {
                    attendanceList = Arrays.asList(optionalAttendance.get());
                    System.out.println("Found record using Optional method");
                } else {
                    // Try to find any records for this employee
                    List<Attendance> allEmployeeRecords = attendanceRepository.findAllByEmployeeId(employeeId);
                    System.out.println("All records for employee " + employeeId + ": " + allEmployeeRecords.size());

                    // Filter for today's date
                    attendanceList = allEmployeeRecords.stream()
                            .filter(a -> a.getDate().equals(today))
                            .collect(Collectors.toList());
                    System.out.println("Filtered records for today: " + attendanceList.size());
                }
            }

        } catch (Exception e) {
            System.err.println("Error finding attendance records: " + e.getMessage());
            e.printStackTrace();
        }

        // ✅ VALIDATION 1: Check if attendance record exists for today
        if (attendanceList == null || attendanceList.isEmpty()) {
            System.out.println("No attendance records found for today");
            return "No check-in found for today. Please check-in first before attempting checkout.";
        }

        Attendance attendance = attendanceList.get(0);
        System.out.println("Found attendance record:");
        System.out.println("  ID: " + attendance.getId());
        System.out.println("  Date: " + attendance.getDate());
        System.out.println("  Employee ID: " + attendance.getEmployeeId());
        System.out.println("  Employee Name: " + attendance.getEmployeeName());
        System.out.println("  Status: " + attendance.getStatus());
        System.out.println("  Check-in Time: " + attendance.getCheckinTime());
        System.out.println("  Check-out Time: " + attendance.getCheckoutTime());

        // ✅ VALIDATION 2: Check if check-in time is marked
        if (attendance.getCheckinTime() == null) {
            System.out.println("Check-in time is null");
            return "Check-in time not found for today. Please complete check-in first.";
        }

        // ✅ VALIDATION 3: Check if status allows checkout - UPDATED FOR "Absent"
        String currentStatus = attendance.getStatus();
        if (currentStatus != null) {
            System.out.println("Original status: " + currentStatus);

            // ✅ UPDATED: Check for "Absent" instead of "On Leave"
            if (currentStatus.equals("Absent")) {
                return "Cannot checkout: Employee is marked as 'Absent' for today. Contact admin if this is incorrect.";
            }
        }

        // ✅ VALIDATION 4: Check if already checked out
        if (attendance.getCheckoutTime() != null) {
            System.out.println("Already checked out at: " + attendance.getCheckoutTime());
            return String.format("Already checked out today at %s. Multiple checkouts are not allowed.",
                    attendance.getCheckoutTime().format(DateTimeFormatter.ofPattern("HH:mm:ss")));
        }

        try {
            LocalTime checkoutTime;
            try {
                // Try parsing as ISO_INSTANT (e.g., 2025-06-17T10:40:15Z)
                checkoutTime = Instant.parse(timeStr)
                        .atZone(ZoneId.systemDefault())
                        .toLocalTime();
                System.out.println("Parsed using Instant: " + checkoutTime);
            } catch (Exception e1) {
                try {
                    // Try parsing as plain time string (e.g., 10:40:15)
                    checkoutTime = LocalTime.parse(timeStr);
                    System.out.println("Parsed using LocalTime: " + checkoutTime);
                } catch (Exception e2) {
                    System.err.println("❌ Failed to parse checkout time: " + timeStr);
                    return "Invalid time format: " + timeStr;
                }
            }

            System.out.println("Parsed checkout time: " + checkoutTime);
            System.out.println("Check-in time: " + attendance.getCheckinTime());

            // ✅ VALIDATION 5: Validate checkout time is after checkin time
            if (checkoutTime.isBefore(attendance.getCheckinTime())) {
                return String.format("Checkout time (%s) cannot be before check-in time (%s)",
                        checkoutTime.format(DateTimeFormatter.ofPattern("HH:mm:ss")),
                        attendance.getCheckinTime().format(DateTimeFormatter.ofPattern("HH:mm:ss")));
            }

            // ✅ VALIDATION 6: Ensure checkout is not in the future (optional)
            LocalTime currentTime = LocalTime.now();
            if (checkoutTime.isAfter(currentTime.plusMinutes(5))) { // Allow 5 minutes buffer
                return "Checkout time cannot be in the future";
            }

            // ✅ All validations passed - proceed with checkout
            attendance.setCheckoutTime(checkoutTime);

            // ✅ UPDATED: Calculate work duration and determine status - Uses "Absent" instead of "On Leave"
            String newStatus = determineCheckoutStatusBasedOnDuration(employeeId, attendance.getCheckinTime(), checkoutTime, currentStatus);
            attendance.setStatus(newStatus);

            System.out.println("Updating attendance with checkout time: " + checkoutTime);
            System.out.println("New status: " + attendance.getStatus());

            Attendance savedAttendance = attendanceRepository.save(attendance);
            System.out.println("Successfully saved checkout. Record ID: " + savedAttendance.getId());

            String workDuration = calculateWorkDurationSummary(attendance.getCheckinTime(), checkoutTime, workTiming);
            System.out.println("Work duration: " + workDuration);

            return String.format("Successfully checked out at %s. %s Status: %s",
                    checkoutTime.format(DateTimeFormatter.ofPattern("HH:mm:ss")),
                    workDuration,
                    newStatus);

        } catch (Exception e) {
            System.err.println("Error processing checkout: " + e.getMessage());
            e.printStackTrace();
            return "Error processing checkout: " + e.getMessage();
        }
    }
    /**
     * ✅ UPDATED: Calculate work duration summary without early leave concept
     */
    private String calculateWorkDurationSummary(LocalTime checkinTime, LocalTime checkoutTime, Map<String, LocalTime> workTiming) {
        if (checkinTime == null || checkoutTime == null) {
            return "Work duration: N/A";
        }

        try {
            long actualMinutes = java.time.Duration.between(checkinTime, checkoutTime).toMinutes();
            long actualHours = actualMinutes / 60;
            long actualRemainingMinutes = actualMinutes % 60;

            return String.format("Work duration: %d hours %d minutes", actualHours, actualRemainingMinutes);

        } catch (Exception e) {
            System.err.println("Error calculating work duration: " + e.getMessage());
            return "Work duration: N/A";
        }
    }

    /**
     * ✅ NEW: Determine checkout status based on work duration and gender-specific rules
     */
    private String determineCheckoutStatusBasedOnDuration(String employeeId, LocalTime checkinTime, LocalTime checkoutTime, String currentStatus) {
        try {
            // Get employee gender
            Optional<Employee> employeeOpt = employeeRepository.findById(employeeId);
            if (!employeeOpt.isPresent()) {
                System.out.println("Employee not found, using current status");
                return currentStatus;
            }

            Employee employee = employeeOpt.get();
            String gender = employee.getGender();

            // Calculate work duration in minutes
            long workDurationMinutes = java.time.Duration.between(checkinTime, checkoutTime).toMinutes();
            double workDurationHours = workDurationMinutes / 60.0;

            System.out.println("=== WORK DURATION ANALYSIS ===");
            System.out.println("Employee: " + employee.getUsername() + " (" + gender + ")");
            System.out.println("Work Duration: " + workDurationHours + " hours (" + workDurationMinutes + " minutes)");

            String finalStatus;

            if ("FEMALE".equalsIgnoreCase(gender) || "F".equalsIgnoreCase(gender)) {
                // Female employee rules: 2.5-3 hours = Half Day, <2.5 hours = Absent
                if (workDurationHours >= 2.5 && workDurationHours <= 3.0) {
                    finalStatus = "Half Day Leave";
                    System.out.println("Female employee worked 2.5-3 hours: Half Day Leave");
                } else if (workDurationHours < 2.5) {
                    finalStatus = "Absent"; // ✅ UPDATED: Use "Absent" instead of "On Leave"
                    System.out.println("Female employee worked <2.5 hours: Marking as Absent");
                } else {
                    // More than 3 hours - keep original status or mark as present
                    finalStatus = currentStatus.equals("Late") ? "Late" : "Present";
                    System.out.println("Female employee worked >3 hours: Keeping status as " + finalStatus);
                }
            } else {
                // Male employee rules: 3.5-4 hours = Half Day, <3.5 hours = Absent
                if (workDurationHours >= 3.5 && workDurationHours <= 4.0) {
                    finalStatus = "Half Day Leave";
                    System.out.println("Male employee worked 3.5-4 hours: Half Day Leave");
                } else if (workDurationHours < 3.5) {
                    finalStatus = "Absent"; // ✅ UPDATED: Use "Absent" instead of "On Leave"
                    System.out.println("Male employee worked <3.5 hours: Marking as Absent");
                } else {
                    // More than 4 hours - keep original status or mark as present
                    finalStatus = currentStatus.equals("Late") ? "Late" : "Present";
                    System.out.println("Male employee worked >4 hours: Keeping status as " + finalStatus);
                }
            }

            System.out.println("Final determined status: " + finalStatus);
            return finalStatus;

        } catch (Exception e) {
            System.err.println("Error determining checkout status: " + e.getMessage());
            return currentStatus; // Fallback to current status
        }
    }

    private String calculateWorkDuration(LocalTime checkinTime, LocalTime checkoutTime) {
        if (checkinTime == null || checkoutTime == null) {
            return "N/A";
        }

        try {
            long minutes = java.time.Duration.between(checkinTime, checkoutTime).toMinutes();
            long hours = minutes / 60;
            long remainingMinutes = minutes % 60;

            return String.format("%d hours %d minutes", hours, remainingMinutes);
        } catch (Exception e) {
            System.err.println("Error calculating work duration: " + e.getMessage());
            return "N/A";
        }
    }

    /**
     * ✅ Enhanced method: Validate checkout eligibility - HANDLES "On Leave" STATUS
     */
    public Map<String, Object> validateCheckoutEligibility(String employeeId) {
        Map<String, Object> response = new HashMap<>();
        LocalDate today = LocalDate.now();

        System.out.println("=== VALIDATE CHECKOUT ELIGIBILITY ===");
        System.out.println("Employee ID: " + employeeId);
        System.out.println("Date: " + today);

        try {
            // Try multiple methods to find attendance
            List<Attendance> attendanceList = null;

            // Primary method
            attendanceList = attendanceRepository.findByEmployeeIdAndDate(employeeId, today);
            System.out.println("Primary method - Records found: " + attendanceList.size());

            // Alternative method if primary fails
            if (attendanceList.isEmpty()) {
                Optional<Attendance> optionalAttendance = attendanceRepository.findAttendanceByEmployeeIdAndDate(employeeId, today);
                if (optionalAttendance.isPresent()) {
                    attendanceList = Arrays.asList(optionalAttendance.get());
                    System.out.println("Alternative method - Found record");
                } else {
                    // Last resort - search all records and filter
                    List<Attendance> allRecords = attendanceRepository.findAllByEmployeeId(employeeId);
                    attendanceList = allRecords.stream()
                            .filter(a -> a.getDate().equals(today))
                            .collect(Collectors.toList());
                    System.out.println("Last resort method - Records found: " + attendanceList.size());
                }
            }

            if (attendanceList.isEmpty()) {
                System.out.println("No attendance records found for today");
                response.put("eligible", false);
                response.put("reason", "No check-in record found for today");
                response.put("message", "Please check-in first before attempting checkout");
                return response;
            }

            Attendance attendance = attendanceList.get(0);
            System.out.println("Validating attendance record:");
            System.out.println("  ID: " + attendance.getId());
            System.out.println("  Status: " + attendance.getStatus());
            System.out.println("  Check-in Time: " + attendance.getCheckinTime());
            System.out.println("  Check-out Time: " + attendance.getCheckoutTime());

            // Check if check-in time exists
            if (attendance.getCheckinTime() == null) {
                System.out.println("Check-in time is null - not eligible");
                response.put("eligible", false);
                response.put("reason", "Check-in time not recorded");
                response.put("message", "Check-in time is missing. Please contact admin.");
                return response;
            }

            // Check status restrictions - employees "Absent" cannot checkout
            String currentStatus = attendance.getStatus();
            if (currentStatus != null) {
                System.out.println("Current status: " + currentStatus);

                if (currentStatus.equals("Absent")) {
                    response.put("eligible", false);
                    response.put("reason", "Employee marked as 'Absent'");
                    response.put("message", "Cannot checkout while on absent status");
                    return response;
                }

                // Allow checkout for other work-related statuses
                // "On Duty", "Permission", "Present", "Late" are all valid for checkout  ✅ UPDATED: Comment
            }

            // Check if already checked out
            if (attendance.getCheckoutTime() != null) {
                System.out.println("Already checked out - not eligible");
                response.put("eligible", false);
                response.put("reason", "Already checked out");
                response.put("message", "Checkout already completed at " +
                        attendance.getCheckoutTime().format(DateTimeFormatter.ofPattern("HH:mm:ss")));
                return response;
            }

            // All validations passed
            System.out.println("All validations passed - eligible for checkout");
            response.put("eligible", true);
            response.put("reason", "Eligible for checkout");
            response.put("message", "Ready for checkout");
            response.put("checkinTime", attendance.getCheckinTime().format(DateTimeFormatter.ofPattern("HH:mm:ss")));
            response.put("currentStatus", attendance.getStatus());
            response.put("employeeName", attendance.getEmployeeName());
            response.put("department", attendance.getDepartment());

            return response;

        } catch (Exception e) {
            System.err.println("Error validating checkout eligibility: " + e.getMessage());
            e.printStackTrace();

            response.put("eligible", false);
            response.put("reason", "System error");
            response.put("message", "Error validating checkout eligibility: " + e.getMessage());
            return response;
        }
    }

    /**
     * ✅ FIXED: getAttendanceSummary - TREATS "On Leave" as "Absent" for summary calculations
     */
    public AttendanceSummary getAttendanceSummary(String employeeId) {
        LocalDate today = LocalDate.now();
        LocalDate firstDayOfMonth = today.withDayOfMonth(1);

        List<Attendance> records = attendanceRepository.findByEmployeeIdAndDateBetween(employeeId, firstDayOfMonth, today);

        AttendanceSummary summary = new AttendanceSummary();
        int present = 0, late = 0, onDuty = 0, absent = 0, permission = 0;

        for (Attendance record : records) {
            if (record.getStatus() != null) {
                String status = record.getStatus();

                switch (status) {
                    case "Present":
                    case "Checked-out":
                        present++;
                        break;
                    case "Late":
                        late++;
                        break;
                    case "On Duty":
                        onDuty++;
                        break;
                    case "Absent":
                        absent++;
                        break;
                    case "Permission":  // ✅ UPDATED: Changed from "Permission Granted"
                        permission++;
                        break;
                    default:
                        // Handle any status that includes "(Checked-out)" suffix
                        if (status.contains("Present") || status.contains("Checked-out")) {
                            present++;
                        } else if (status.contains("Late")) {
                            late++;
                        } else if (status.contains("On Duty")) {
                            onDuty++;
                        } else if (status.contains("Permission")) {  // ✅ UPDATED: Changed from "Permission Granted"
                            permission++;
                        }
                        break;
                }
            }
        }

        int totalDays = today.getDayOfMonth();
        int totalAbsent = totalDays - (present + late + onDuty + permission) + absent;

        // Get today's status
        String todayStatus = "Absent";
        Optional<Attendance> todayRecord = records.stream()
                .filter(r -> r.getDate().equals(today))
                .findFirst();

        if (todayRecord.isPresent()) {
            todayStatus = todayRecord.get().getStatus();
        }

        summary.setTodayStatus(todayStatus);
        summary.setMonthlyPresent(present);
        summary.setMonthlyAbsent(totalAbsent);
        summary.setMonthlyLate(late);
        summary.setMonthlyPermission(onDuty + permission);

        int totalAttended = present + late + onDuty + permission;
        summary.setMonthlyAverage(totalDays > 0 ? (totalAttended * 100 / totalDays) + "%" : "0%");
        summary.setOverallRate(totalDays > 0 ? (totalAttended * 100 / totalDays) + "%" : "0%");

        System.out.println("=== ATTENDANCE SUMMARY DEBUG ===");
        System.out.println("Employee ID: " + employeeId);
        System.out.println("Total Records: " + records.size());
        System.out.println("Present: " + present);
        System.out.println("Late: " + late);
        System.out.println("On Duty: " + onDuty);
        System.out.println("Absent: " + absent);
        System.out.println("Permission: " + permission);
        System.out.println("Total Absent: " + totalAbsent);
        System.out.println("Today Status: " + todayStatus);

        return summary;
    }

    /**
     * ✅ FIXED: getAdminDashboardStats - TREATS "On Leave" as "Absent"
     */
    public AdminDashboardStats getAdminDashboardStats() {
        LocalDate today = LocalDate.now();
        List<Attendance> todaysAttendance = attendanceRepository.findByDate(today);
        long totalEmployees = employeeRepository.count();

        long presentToday = 0;
        long lateArrivals = 0;
        long onDuty = 0;
        long permissionGranted = 0;  // Keep variable name for clarity
        long absentCount = 0;

        // Count each status using exact matching
        for (Attendance attendance : todaysAttendance) {
            if (attendance.getStatus() != null) {
                String status = attendance.getStatus();

                switch (status) {
                    case "Present":
                    case "Checked-out":
                        presentToday++;
                        break;
                    case "Late":
                        lateArrivals++;
                        break;
                    case "On Duty":
                        onDuty++;
                        break;
                    case "Absent":
                        absentCount++;
                        break;
                    case "Permission":  // ✅ UPDATED: Changed from "Permission Granted"
                        permissionGranted++;
                        break;
                    default:
                        // Handle statuses with "(Checked-out)" suffix
                        if (status.contains("Present") || status.contains("Checked-out")) {
                            presentToday++;
                        } else if (status.contains("Late")) {
                            lateArrivals++;
                        } else if (status.contains("On Duty")) {
                            onDuty++;
                        } else if (status.contains("Permission")) {  // ✅ UPDATED: Changed from "Permission Granted"
                            permissionGranted++;
                        } else {
                            // Log unknown statuses for debugging
                            System.out.println("Unknown status found: " + status);
                        }
                        break;
                }
            }
        }

        // Calculate absent employees
        long totalPresent = presentToday + lateArrivals + onDuty + permissionGranted;
        long absent = totalEmployees - totalPresent;

        // Calculate attendance rate
        String rate = totalEmployees == 0 ? "0%" :
                String.format("%.1f%%", (totalPresent * 100.0 / totalEmployees));

        AdminDashboardStats stats = new AdminDashboardStats();
        stats.setTotalEmployees(totalEmployees);
        stats.setPresentToday(presentToday);
        stats.setOnDuty(onDuty);
        stats.setAbsent(absent);
        stats.setLateArrivals(lateArrivals);
        stats.setPermissionCount(permissionGranted);
        stats.setAttendanceRate(rate);

        // Debug logging
        System.out.println("=== DASHBOARD STATS DEBUG ===");
        System.out.println("Total Employees: " + totalEmployees);
        System.out.println("Today's Attendance Records: " + todaysAttendance.size());
        System.out.println("Present: " + presentToday);
        System.out.println("Late: " + lateArrivals);
        System.out.println("On Duty: " + onDuty);
        System.out.println("Permission: " + permissionGranted);  // ✅ UPDATED: Log message
        System.out.println("Absent (direct count): " + absentCount);
        System.out.println("Total Absent (calculated): " + absent);
        System.out.println("Attendance Rate: " + rate);

        return stats;
    }

    /**
     * ✅ FIXED: getAllEmployeeStatusToday - CONVERTS "On Leave" to "Absent" for display
     */
    /**
     * ✅ FIXED: getAllEmployeeStatusToday - CONVERTS "On Leave" to "Absent" for display
     */
    public List<EmployeeStatus> getAllEmployeeStatusToday() {
        LocalDate today = LocalDate.now();
        List<Employee> allEmployees = employeeRepository.findAll();
        List<Attendance> todayAttendance = attendanceRepository.findByDate(today);

        Map<String, Attendance> attendanceMap = todayAttendance.stream()
                .collect(Collectors.toMap(Attendance::getEmployeeId, a -> a));

        List<EmployeeStatus> statusList = new ArrayList<>();

        for (Employee emp : allEmployees) {
            String status = "Absent";
            Attendance att = attendanceMap.get(emp.getId());

            if (att != null) {
                status = att.getStatus(); // ✅ No conversion needed since we use "Absent" directly
            }

            statusList.add(new EmployeeStatus(
                    emp.getId(),
                    emp.getUsername(),
                    emp.getDepartment(),
                    emp.getDesignation(),
                    status
            ));
        }

        System.out.println("=== getAllEmployeeStatusToday DEBUG ===");
        System.out.println("Total employees: " + allEmployees.size());
        System.out.println("Today's attendance records: " + todayAttendance.size());

        return statusList;
    }
    public List<EmployeeAttendanceDTO> getTodayEmployeeAttendance() {
        return repository.findTodayEmployeeAttendance();
    }

    public List<EmployeeAttendanceDTO> getAttendanceReport() {
        return repository.findAllEmployeeAttendance();
    }

    /**
     * ✅ REMOVED: normalizeStatus method that was causing the issue
     * Now we preserve the original status throughout the system
     */

    public long getAttendanceCount(String employeeId, LocalDate date) {
        return attendanceRepository.countByEmployeeIdAndDate(employeeId, date);
    }

    public Optional<Attendance> getLatestAttendance(String employeeId) {
        List<Attendance> records = attendanceRepository.findLatestByEmployeeId(employeeId);
        return records.isEmpty() ? Optional.empty() : Optional.of(records.get(0));
    }
    /**
     * ✅ NEW: Get employee expected work hours based on gender
     */
    // Updated getEmployeeWorkTiming method
    private Map<String, LocalTime> getEmployeeWorkTiming(String employeeId) {
        Map<String, LocalTime> timing = new HashMap<>();

        try {
            Optional<Employee> employeeOpt = employeeRepository.findById(employeeId);
            if (employeeOpt.isPresent()) {
                Employee employee = employeeOpt.get();
                String gender = employee.getGender();

                System.out.println("Employee: " + employee.getUsername() + ", Gender: " + gender);

                // Special timing for mmw027
                if ("mmw027".equals(employeeId)) {
                    timing.put("startTime", LocalTime.of(9, 40)); // 9:40 AM
                    timing.put("endTime", LocalTime.of(16, 0));   // 4:00 PM
                    timing.put("graceEndTime", LocalTime.of(9, 40)); // No grace period for special employee
                    System.out.println("Special employee mmw027 - Work timing: 9:40 AM to 4:00 PM (No grace period)");
                    return timing;
                }

                // Standard timing for all other employees
                timing.put("startTime", LocalTime.of(9, 0)); // 9:00 AM official start
                timing.put("graceEndTime", LocalTime.of(9, 30)); // Grace period ends at 9:30 AM

                // Different end times based on gender
                if ("FEMALE".equalsIgnoreCase(gender) || "F".equalsIgnoreCase(gender)) {
                    timing.put("endTime", LocalTime.of(17, 0)); // 5:00 PM for females
                    System.out.println("Female employee - Work timing: 9:00 AM to 5:00 PM (Grace until 9:30 AM)");
                } else {
                    timing.put("endTime", LocalTime.of(18, 0)); // 6:00 PM for males
                    System.out.println("Male employee - Work timing: 9:00 AM to 6:00 PM (Grace until 9:30 AM)");
                }
            } else {
                // Default timing if employee not found
                timing.put("startTime", LocalTime.of(9, 0));
                timing.put("endTime", LocalTime.of(18, 0));
                timing.put("graceEndTime", LocalTime.of(9, 30));
                System.out.println("Employee not found, using default timing");
            }
        } catch (Exception e) {
            System.err.println("Error getting employee work timing: " + e.getMessage());
            // Default timing in case of error
            timing.put("startTime", LocalTime.of(9, 0));
            timing.put("endTime", LocalTime.of(18, 0));
            timing.put("graceEndTime", LocalTime.of(9, 30));
        }

        return timing;
    }

    // Updated determineAttendanceStatusWithGender method
    private String determineAttendanceStatusWithGender(String employeeId, LocalTime checkinTime) {
        try {
            // Get employee-specific work timing based on gender and special cases
            Map<String, LocalTime> workTiming = getEmployeeWorkTiming(employeeId);
            LocalTime officialStartTime = workTiming.get("startTime");
            LocalTime graceEndTime = workTiming.get("graceEndTime");

            System.out.println("Official start time: " + officialStartTime);
            System.out.println("Grace end time: " + graceEndTime);
            System.out.println("Actual checkin time: " + checkinTime);

            // Check if within grace period or on time
            if (checkinTime.isBefore(graceEndTime) || checkinTime.equals(graceEndTime)) {
                System.out.println("Employee is within grace period - marking as Present");
                return "Present";
            } else {
                // Late arrival - need to check employee status before marking as Late
                System.out.println("Employee is late (after grace period) - checking current status");

                // Check current employee status for today
                String currentStatus = getCurrentEmployeeStatus(employeeId);
                System.out.println("Current employee status: " + currentStatus);

                // Only mark as Late if current status is Absent or Permission
                if ("Absent".equalsIgnoreCase(currentStatus) || "Permission".equalsIgnoreCase(currentStatus)) {
                    System.out.println("Employee status allows late marking - marking as Late");
                    return "Late";
                } else {
                    System.out.println("Employee status does not allow late marking - marking as Present");
                    return "Present";
                }
            }

        } catch (Exception e) {
            System.err.println("Error determining attendance status: " + e.getMessage());
            // Fallback to default logic
            return "Present"; // Default to Present in case of error
        }
    }

    // New method to check current employee status
    private String getCurrentEmployeeStatus(String employeeId) {
        try {
            LocalDate today = LocalDate.now();

            // Check if there's already an attendance record for today
            List<Attendance> existingAttendanceList = attendanceRepository
                    .findByEmployeeIdAndDate(employeeId, today);

            if (!existingAttendanceList.isEmpty()) {
                // Get the first (or most recent) attendance record
                Attendance existingAttendance = existingAttendanceList.get(0);
                String currentStatus = existingAttendance.getStatus();
                System.out.println("Found existing attendance record with status: " + currentStatus);
                return currentStatus;
            }

            // If no attendance record exists, check if employee was marked as Absent or Permission
            // This could be from manual entry or other systems
            // You might need to adjust this based on your database structure

            // For now, return "Absent" as default for employees without attendance record
            // This allows late marking for employees who haven't been marked present yet
            System.out.println("No existing attendance record found - defaulting to Absent");
            return "Absent";

        } catch (Exception e) {
            System.err.println("Error checking current employee status: " + e.getMessage());
            return "Absent"; // Default to allow late marking
        }
    }

    private String determineCheckoutStatus(String currentStatus, LocalTime checkoutTime, LocalTime expectedEndTime) {
        if (checkoutTime.isBefore(expectedEndTime)) {
            return currentStatus + " (Early Leave)";
        } else if (checkoutTime.equals(expectedEndTime) || checkoutTime.isBefore(expectedEndTime.plusMinutes(30))) {
            return currentStatus + " (On Time)";
        } else {
            return currentStatus + " (Overtime)";
        }
    }

    /**
     * ✅ UPDATED: Calculate work duration with expected hours comparison
     */
    // Updated calculateWorkDurationWithExpected method (minor fix for expected duration calculation)
    private String calculateWorkDurationWithExpected(LocalTime checkinTime, LocalTime checkoutTime, Map<String, LocalTime> workTiming) {
        if (checkinTime == null || checkoutTime == null) {
            return "Work duration: N/A";
        }

        try {
            long actualMinutes = java.time.Duration.between(checkinTime, checkoutTime).toMinutes();
            long actualHours = actualMinutes / 60;
            long actualRemainingMinutes = actualMinutes % 60;

            // Calculate expected work duration using the correct start and end times
            LocalTime expectedStart = workTiming.get("startTime");
            LocalTime expectedEnd = workTiming.get("endTime");
            long expectedMinutes = java.time.Duration.between(expectedStart, expectedEnd).toMinutes();
            long expectedHours = expectedMinutes / 60;

            String actualDuration = String.format("Total work duration: %d hours %d minutes", actualHours, actualRemainingMinutes);
            String expectedDuration = String.format("Expected: %d hours", expectedHours);

            if (actualMinutes < expectedMinutes) {
                long shortfallMinutes = expectedMinutes - actualMinutes;
                long shortfallHours = shortfallMinutes / 60;
                long shortfallRemainingMinutes = shortfallMinutes % 60;
                return String.format("%s, %s (Short by %d hours %d minutes)",
                        actualDuration, expectedDuration, shortfallHours, shortfallRemainingMinutes);
            } else if (actualMinutes > expectedMinutes) {
                long overtimeMinutes = actualMinutes - expectedMinutes;
                long overtimeHours = overtimeMinutes / 60;
                long overtimeRemainingMinutes = overtimeMinutes % 60;
                return String.format("%s, %s (Overtime: %d hours %d minutes)",
                        actualDuration, expectedDuration, overtimeHours, overtimeRemainingMinutes);
            } else {
                return String.format("%s, %s (Perfect attendance)", actualDuration, expectedDuration);
            }

        } catch (Exception e) {
            System.err.println("Error calculating work duration: " + e.getMessage());
            return "Work duration: N/A";
        }
    }
    public boolean validateAllocationTiming(String allocationType, LocalTime expectedStart,
                                            LocalTime expectedEnd, LocalTime actualTime) {

        if ("Permission".equals(allocationType)) {
            // Permission: Must be within allocated time window
            return !actualTime.isBefore(expectedStart) && !actualTime.isAfter(expectedEnd);

        } else if ("Half Day".equals(allocationType)) {
            // Half Day: Must be within allocated time window
            return !actualTime.isBefore(expectedStart) && !actualTime.isAfter(expectedEnd);
        }

        return false;
    }
    public String determineConvertedStatus(String originalType, boolean isLate) {
        if (!isLate) {
            return originalType; // No conversion needed
        }

        if ("Permission".equals(originalType)) {
            return "Half Day"; // Late permission becomes half day
        } else if ("Half Day".equals(originalType)) {
            return "Absent"; // ✅ UPDATED: Late half day becomes absent (instead of "On Leave")
        }

        return originalType;
    }

    public Map<String, Object> validateSmartTiming(String allocationType, String startTimeStr,
                                                   String endTimeStr, LocalTime currentTime) {
        Map<String, Object> result = new HashMap<>();

        try {
            LocalTime startTime = LocalTime.parse(startTimeStr);
            LocalTime endTime = LocalTime.parse(endTimeStr);

            boolean isWithinTime = !currentTime.isBefore(startTime) && !currentTime.isAfter(endTime);
            boolean isEarly = currentTime.isBefore(startTime);
            boolean isLate = currentTime.isAfter(endTime);

            result.put("isWithinTime", isWithinTime);
            result.put("isEarly", isEarly);
            result.put("isLate", isLate);

            if (isWithinTime || isEarly) {
                result.put("validationStatus", "VALID");
                result.put("message", "Employee arrived within allocated time");
                result.put("finalStatus", allocationType);
            } else if (isLate) {
                String convertedStatus = determineConvertedStatus(allocationType, true);
                result.put("validationStatus", "LATE_CONVERTED");
                result.put("message", String.format("Late arrival - Converting %s to %s",
                        allocationType, convertedStatus));
                result.put("finalStatus", convertedStatus);
            }

            // Calculate time difference
            if (isLate) {
                long minutesLate = java.time.Duration.between(endTime, currentTime).toMinutes();
                result.put("minutesLate", minutesLate);
            } else if (isEarly) {
                long minutesEarly = java.time.Duration.between(currentTime, startTime).toMinutes();
                result.put("minutesEarly", minutesEarly);
            }

        } catch (Exception e) {
            result.put("validationStatus", "ERROR");
            result.put("message", "Error validating timing: " + e.getMessage());
            result.put("finalStatus", allocationType);
        }

        return result;
    }

    /**
     * Get today's attendance for all employees
     */
    public Map<String, Object> getTodayAttendanceForAll() {
        LocalDate today = LocalDate.now();
        List<Attendance> todayAttendance = attendanceRepository.findByDate(today);

        Map<String, Object> result = new HashMap<>();
        Map<String, String> employeeStatus = new HashMap<>();
        Map<String, Long> statusCounts = new HashMap<>();

        // Initialize status counts
        statusCounts.put("present", 0L);
        statusCounts.put("absent", 0L);
        statusCounts.put("late", 0L);
        statusCounts.put("onduty", 0L);
        statusCounts.put("halfday", 0L);
        statusCounts.put("permission", 0L);

        // Process attendance records
        for (Attendance attendance : todayAttendance) {
            employeeStatus.put(attendance.getEmployeeId(), attendance.getStatus());
            statusCounts.put(attendance.getStatus(), statusCounts.get(attendance.getStatus()) + 1);
        }

        result.put("employeeStatus", employeeStatus);
        result.put("statusCounts", statusCounts);
        result.put("totalRecords", todayAttendance.size());

        return result;
    }
    /**
     * Get attendance trends for analytics
     */
    public Map<String, Object> getAttendanceTrends(String employeeId, int days) {
        LocalDate endDate = LocalDate.now();
        LocalDate startDate = endDate.minus(days, ChronoUnit.DAYS);

        List<Attendance> attendanceList = attendanceRepository.findByEmployeeIdAndDateBetweenOrderByDateAsc(employeeId, startDate, endDate);

        Map<String, Object> trends = new HashMap<>();
        Map<String, Integer> dailyStatus = new HashMap<>();

        for (Attendance attendance : attendanceList) {
            String dateKey = attendance.getDate().toString();
            switch (attendance.getStatus().toLowerCase()) {
                case "present":
                case "onduty":
                    dailyStatus.put(dateKey, 1);
                    break;
                case "late":
                case "halfday":
                case "permission":
                    dailyStatus.put(dateKey, 0);
                    break;
                default:
                    dailyStatus.put(dateKey, -1);
            }
        }

        trends.put("dailyStatus", dailyStatus);
        trends.put("period", days);
        trends.put("startDate", startDate);
        trends.put("endDate", endDate);

        return trends;
    }
    /**
     * Check if employee has attendance record for today
     */
    public boolean hasAttendanceToday(String employeeId) {
        LocalDate today = LocalDate.now();
        return attendanceRepository.existsByEmployeeIdAndDate(employeeId, today);
    }

    /**
     * 21/6/25
     * ✅ NEW: Validate permission request based on monthly limits and rules
     */
    public Map<String, Object> validatePermissionRequest(String employeeId, LocalTime permissionFrom,
                                                         LocalTime permissionTo, LocalDate requestDate) {
        Map<String, Object> result = new HashMap<>();

        try {
            System.out.println("=== VALIDATING PERMISSION REQUEST ===");
            System.out.println("Employee ID: " + employeeId);
            System.out.println("Permission From: " + permissionFrom);
            System.out.println("Permission To: " + permissionTo);
            System.out.println("Request Date: " + requestDate);

            // Calculate requested permission duration in minutes
            long requestedMinutes = java.time.Duration.between(permissionFrom, permissionTo).toMinutes();
            System.out.println("Requested duration: " + requestedMinutes + " minutes");

            // Rule 1: Minimum 1 hour (60 minutes) required
            if (requestedMinutes < 60) {
                result.put("valid", false);
                result.put("reason", "MINIMUM_TIME_NOT_MET");
                result.put("message", String.format("Permission cannot be granted for less than 1 hour. Requested: %d minutes (minimum 60 minutes required)", requestedMinutes));
                return result;
            }

            // Get current month tracker
            String currentYearMonth = YearMonth.from(requestDate).toString();
            MonthlyPermissionTracker tracker = getOrCreatePermissionTracker(employeeId, currentYearMonth);

            int usedMinutes = tracker.getTotalPermissionMinutesUsed();
            int remainingMinutes = 120 - usedMinutes; // 2 hours total limit
            int permissionCount = tracker.getPermissionCount();

            System.out.println("Current month tracker:");
            System.out.println("  Total minutes used: " + usedMinutes);
            System.out.println("  Remaining minutes: " + remainingMinutes);
            System.out.println("  Permission count: " + permissionCount);

            // Rule 2: Check if monthly 2-hour limit already exhausted
            if (usedMinutes >= 120) {
                result.put("valid", false);
                result.put("reason", "MONTHLY_LIMIT_EXHAUSTED");
                result.put("message", "Permission already over. Monthly 2-hour permission limit has been completely exhausted.");
                return result;
            }

            // Rule 3: Maximum 2 permissions per month
            if (permissionCount >= 2) {
                result.put("valid", false);
                result.put("reason", "PERMISSION_COUNT_EXCEEDED");
                result.put("message", "Permission already over. Maximum 2 permissions per month have been used.");
                return result;
            }

            // Rule 4: Check if requested time exceeds remaining monthly allowance
            if (requestedMinutes > remainingMinutes) {
                result.put("valid", false);
                result.put("reason", "EXCEEDS_REMAINING_TIME");
                result.put("message", String.format("Cannot grant %d minutes permission. Only %d minutes remaining from monthly 2-hour limit. Please adjust the time range.",
                        requestedMinutes, remainingMinutes));
                return result;
            }

            // Rule 5: Check minimum remaining time after this permission
            if (remainingMinutes - requestedMinutes > 0 && remainingMinutes - requestedMinutes < 60 && permissionCount == 0) {
                // If this is the first permission and will leave less than 60 minutes for second permission
                // Allow it but warn that second permission might not be possible
                System.out.println("Warning: This permission will leave less than 60 minutes for future permissions");
            }

            // Permission request is valid
            result.put("valid", true);
            result.put("reason", "VALID_REQUEST");
            result.put("message", String.format("Permission request approved. %d minutes will be allocated from your monthly allowance.", requestedMinutes));
            result.put("requestedMinutes", requestedMinutes);
            result.put("remainingMinutesAfter", remainingMinutes - requestedMinutes);
            result.put("permissionCountAfter", permissionCount + 1);
            result.put("currentUsedMinutes", usedMinutes);
            result.put("currentRemainingMinutes", remainingMinutes);

            return result;

        } catch (Exception e) {
            System.err.println("Error validating permission request: " + e.getMessage());
            e.printStackTrace();

            result.put("valid", false);
            result.put("reason", "SYSTEM_ERROR");
            result.put("message", "Error validating permission request: " + e.getMessage());
            return result;
        }
    }

    /**
     * 21/6/25
     * ✅ NEW: Process permission checkout with grace time and conversion rules
     */
    public Map<String, Object> processPermissionCheckout(String employeeId, LocalTime actualCheckoutTime,
                                                         LocalDate date, LocalTime permissionFrom, LocalTime permissionTo) {
        Map<String, Object> result = new HashMap<>();

        try {
            System.out.println("=== PROCESSING PERMISSION CHECKOUT ===");
            System.out.println("Employee ID: " + employeeId);
            System.out.println("Actual Checkout: " + actualCheckoutTime);
            System.out.println("Permission From: " + permissionFrom);
            System.out.println("Permission To: " + permissionTo);

            // Calculate allocated permission duration
            long allocatedMinutes = java.time.Duration.between(permissionFrom, permissionTo).toMinutes();

            // Calculate actual permission duration (from permission start to actual checkout)
            long actualMinutes = java.time.Duration.between(permissionFrom, actualCheckoutTime).toMinutes();

            System.out.println("Allocated minutes: " + allocatedMinutes);
            System.out.println("Actual minutes: " + actualMinutes);

            // Get current month tracker
            String currentYearMonth = YearMonth.from(date).toString();
            MonthlyPermissionTracker tracker = getOrCreatePermissionTracker(employeeId, currentYearMonth);

            // Determine which permission this is (first or second)
            boolean isFirstPermission = tracker.getPermissionCount() == 0;
            boolean canUseGrace = isFirstPermission ? !tracker.isFirstPermissionGraceUsed() : !tracker.isSecondPermissionGraceUsed();

            System.out.println("Is first permission: " + isFirstPermission);
            System.out.println("Can use grace: " + canUseGrace);

            // Calculate grace time available
            int graceMinutesAvailable = 0;
            if (canUseGrace) {
                if (allocatedMinutes == 120) {
                    // Single 2-hour permission gets 5 minutes grace total
                    graceMinutesAvailable = 5;
                } else {
                    // Each 1-hour permission gets 5 minutes grace
                    graceMinutesAvailable = 5;
                }
            }

            System.out.println("Grace minutes available: " + graceMinutesAvailable);

            // Check if within grace time
            boolean withinGraceTime = actualMinutes <= (allocatedMinutes + graceMinutesAvailable);

            String finalStatus;
            String message;

            if (actualCheckoutTime.isBefore(permissionTo)) {
                // Checked out early - no issue
                finalStatus = "Permission";
                message = "Permission completed successfully (early checkout)";

                // Update tracker
                updatePermissionTracker(tracker, (int) allocatedMinutes, isFirstPermission, false);

            } else if (withinGraceTime) {
                // Within grace time - allow as permission
                finalStatus = "Permission";
                message = String.format("Permission completed with %d minutes grace time used",
                        actualMinutes - allocatedMinutes);

                // Update tracker with grace used
                updatePermissionTracker(tracker, (int) actualMinutes, isFirstPermission, true);

            } else {
                // Exceeded grace time - convert to half day and consume remaining permission time
                finalStatus = "Half Day";

                // Calculate how much extra time was taken
                long extraMinutes = actualMinutes - allocatedMinutes - graceMinutesAvailable;

                // If this exceeds the remaining monthly permission, it becomes half day
                int remainingPermissionMinutes = 120 - tracker.getTotalPermissionMinutesUsed();

                if (actualMinutes > remainingPermissionMinutes) {
                    // Consume all remaining permission time
                    updatePermissionTrackerForExcess(tracker, remainingPermissionMinutes, isFirstPermission);
                    message = String.format("Permission exceeded grace time by %d minutes. Status changed to Half Day. All remaining permission time (%d minutes) consumed.",
                            extraMinutes, remainingPermissionMinutes);
                } else {
                    // Consume the actual time taken
                    updatePermissionTrackerForExcess(tracker, (int) actualMinutes, isFirstPermission);
                    message = String.format("Permission exceeded grace time by %d minutes. Status changed to Half Day. %d minutes consumed from monthly allowance.",
                            extraMinutes, actualMinutes);
                }
            }

            // Save the updated tracker
            permissionTrackerRepository.save(tracker);

            result.put("finalStatus", finalStatus);
            result.put("message", message);
            result.put("allocatedMinutes", allocatedMinutes);
            result.put("actualMinutes", actualMinutes);
            result.put("graceUsed", !withinGraceTime ? 0 : Math.max(0, actualMinutes - allocatedMinutes));
            result.put("withinGraceTime", withinGraceTime);
            result.put("remainingPermissionMinutes", 120 - tracker.getTotalPermissionMinutesUsed());
            result.put("remainingPermissionCount", 2 - tracker.getPermissionCount());

            System.out.println("Final status: " + finalStatus);
            System.out.println("Remaining permission minutes: " + (120 - tracker.getTotalPermissionMinutesUsed()));

            return result;

        } catch (Exception e) {
            System.err.println("Error processing permission checkout: " + e.getMessage());
            e.printStackTrace();

            result.put("finalStatus", "Permission");
            result.put("message", "Error processing permission checkout: " + e.getMessage());
            return result;
        }
    }

    /**
     * 21/6/25
     * ✅ NEW: Helper method to get or create permission tracker
     */
    private MonthlyPermissionTracker getOrCreatePermissionTracker(String employeeId, String yearMonth) {
        Optional<MonthlyPermissionTracker> existingTracker =
                permissionTrackerRepository.findByEmployeeIdAndYearMonth(employeeId, yearMonth);

        if (existingTracker.isPresent()) {
            return existingTracker.get();
        } else {
            MonthlyPermissionTracker newTracker = new MonthlyPermissionTracker(employeeId, yearMonth);
            return permissionTrackerRepository.save(newTracker);
        }
    }

    /**
     * 21/6/25
     * ✅ NEW: Update permission tracker for normal permission usage
     */
    private void updatePermissionTracker(MonthlyPermissionTracker tracker, int minutesUsed,
                                         boolean isFirstPermission, boolean graceUsed) {
        tracker.setTotalPermissionMinutesUsed(tracker.getTotalPermissionMinutesUsed() + minutesUsed);
        tracker.setPermissionCount(tracker.getPermissionCount() + 1);

        if (graceUsed) {
            if (isFirstPermission) {
                tracker.setFirstPermissionGraceUsed(true);
            } else {
                tracker.setSecondPermissionGraceUsed(true);
            }
            tracker.setGraceMinutesUsed(tracker.getGraceMinutesUsed() + (minutesUsed - (isFirstPermission ? 60 : (120 - tracker.getTotalPermissionMinutesUsed() + minutesUsed))));
        }

        System.out.println("Updated tracker - Total minutes: " + tracker.getTotalPermissionMinutesUsed() +
                ", Count: " + tracker.getPermissionCount() +
                ", Grace used: " + graceUsed);
    }

    /**
     * 21/6/25
     * ✅ NEW: Update permission tracker when permission exceeds limits
     */
    private void updatePermissionTrackerForExcess(MonthlyPermissionTracker tracker, int minutesConsumed,
                                                  boolean isFirstPermission) {
        tracker.setTotalPermissionMinutesUsed(tracker.getTotalPermissionMinutesUsed() + minutesConsumed);

        // If this exceeds or equals the monthly limit, mark both permissions as used
        if (tracker.getTotalPermissionMinutesUsed() >= 120) {
            tracker.setPermissionCount(2); // Max out the count
            tracker.setFirstPermissionGraceUsed(true);
            tracker.setSecondPermissionGraceUsed(true);
        } else {
            tracker.setPermissionCount(tracker.getPermissionCount() + 1);
            if (isFirstPermission) {
                tracker.setFirstPermissionGraceUsed(true);
            } else {
                tracker.setSecondPermissionGraceUsed(true);
            }
        }

        System.out.println("Updated tracker for excess - Total minutes: " + tracker.getTotalPermissionMinutesUsed() +
                ", Count: " + tracker.getPermissionCount());
    }
    /**
     * 21/6/25
     * ✅ NEW: Get monthly permission summary for employee
     */
    public Map<String, Object> getMonthlyPermissionSummary(String employeeId, YearMonth yearMonth) {
        Map<String, Object> summary = new HashMap<>();

        try {
            String yearMonthStr = yearMonth.toString();
            Optional<MonthlyPermissionTracker> trackerOpt =
                    permissionTrackerRepository.findByEmployeeIdAndYearMonth(employeeId, yearMonthStr);

            if (trackerOpt.isPresent()) {
                MonthlyPermissionTracker tracker = trackerOpt.get();

                summary.put("totalMinutesUsed", tracker.getTotalPermissionMinutesUsed());
                summary.put("totalMinutesRemaining", 120 - tracker.getTotalPermissionMinutesUsed());
                summary.put("permissionCount", tracker.getPermissionCount());
                summary.put("permissionRemaining", 2 - tracker.getPermissionCount());
                summary.put("graceMinutesUsed", tracker.getGraceMinutesUsed());
                summary.put("firstPermissionGraceUsed", tracker.isFirstPermissionGraceUsed());
                summary.put("secondPermissionGraceUsed", tracker.isSecondPermissionGraceUsed());

                // Calculate available grace for future permissions
                int availableGrace = 0;
                if (!tracker.isFirstPermissionGraceUsed() && tracker.getPermissionCount() < 1) {
                    availableGrace += 5;
                }
                if (!tracker.isSecondPermissionGraceUsed() && tracker.getPermissionCount() < 2) {
                    availableGrace += 5;
                }
                summary.put("availableGraceMinutes", availableGrace);

            } else {
                // No permissions used this month
                summary.put("totalMinutesUsed", 0);
                summary.put("totalMinutesRemaining", 120);
                summary.put("permissionCount", 0);
                summary.put("permissionRemaining", 2);
                summary.put("graceMinutesUsed", 0);
                summary.put("firstPermissionGraceUsed", false);
                summary.put("secondPermissionGraceUsed", false);
                summary.put("availableGraceMinutes", 10); // 5 + 5 for both permissions
            }

            summary.put("yearMonth", yearMonthStr);
            summary.put("employeeId", employeeId);

        } catch (Exception e) {
            System.err.println("Error getting monthly permission summary: " + e.getMessage());
            summary.put("error", "Error retrieving permission summary: " + e.getMessage());
        }

        return summary;
    }

    /**
     * 21/6/25
     * ✅ UPDATED: Enhanced checkIn method with permission validation
     */
    public Attendance checkInWithPermission(Map<String, String> body) {
        String employeeId = body.get("employeeId");
        String time = body.get("time");
        String permissionFromStr = body.get("permissionFrom");
        String permissionToStr = body.get("permissionTo");

        System.out.println("=== CHECK-IN WITH PERMISSION DEBUG ===");
        System.out.println("Employee ID: " + employeeId);
        System.out.println("Time: " + time);
        System.out.println("Permission From: " + permissionFromStr);
        System.out.println("Permission To: " + permissionToStr);

        if (employeeId == null || time == null) {
            throw new IllegalArgumentException("Missing employeeId or time");
        }

        // Validate employee exists
        Employee employee = employeeRepository.findById(employeeId)
                .orElseThrow(() -> new IllegalArgumentException("Employee not found"));

        LocalDate today = LocalDate.now();

        // Check if attendance already exists for today
        List<Attendance> existingRecords = attendanceRepository.findByEmployeeIdAndDate(employeeId, today);
        if (!existingRecords.isEmpty()) {
            Attendance existingAttendance = existingRecords.get(0);
            String displayStatus = getDisplayStatus(existingAttendance.getStatus());
            throw new IllegalArgumentException(
                    String.format("Attendance already registered for today. Current status: %s, Check-in time: %s",
                            displayStatus,
                            existingAttendance.getCheckinTime() != null ? existingAttendance.getCheckinTime().toString() : "Not available")
            );
        }

        LocalTime checkinTime = LocalTime.now().withNano(0);
        String status;

        // Check if this is a permission request
        if (permissionFromStr != null && permissionToStr != null) {
            try {
                LocalTime permissionFrom = LocalTime.parse(permissionFromStr);
                LocalTime permissionTo = LocalTime.parse(permissionToStr);

                // Validate permission request
                Map<String, Object> validationResult = validatePermissionRequest(employeeId, permissionFrom, permissionTo, today);

                if (!(Boolean) validationResult.get("valid")) {
                    throw new IllegalArgumentException("Permission request invalid: " + validationResult.get("message"));
                }

                // Check if employee is within permission time
                if (checkinTime.isAfter(permissionFrom) && checkinTime.isBefore(permissionTo.plusMinutes(5))) {
                    status = "Permission";
                } else {
                    // Employee is outside permission time - determine based on normal rules
                    status = determineAttendanceStatusWithGender(employeeId, checkinTime);
                }

            } catch (Exception e) {
                System.err.println("Error processing permission: " + e.getMessage());
                status = determineAttendanceStatusWithGender(employeeId, checkinTime);
                permissionFromStr = null;
                permissionToStr = null;
            }
        } else {
            // Normal check-in without permission
            status = determineAttendanceStatusWithGender(employeeId, checkinTime);
        }

        Attendance attendance = new Attendance();
        attendance.setEmployeeId(employeeId);
        attendance.setEmployeeName(employee.getUsername());
        attendance.setDepartment(employee.getDepartment());
        attendance.setDesignation(employee.getDesignation());
        attendance.setDate(today);
        attendance.setCheckinTime(checkinTime);
        attendance.setStatus(status);

        // Set permission times if provided
        if (permissionFromStr != null && permissionToStr != null) {
            attendance.setPermissionFrom(LocalTime.parse(permissionFromStr));
            attendance.setPermissionTo(LocalTime.parse(permissionToStr));
        }

        Attendance saved = attendanceRepository.save(attendance);
        System.out.println("Saved attendance record with ID: " + saved.getId() + ", Status: " + status);

        return saved;
    }

    /**
     * 21/6/25
     * ✅ UPDATED: Enhanced recordCheckout method with permission processing
     */
    public String recordCheckoutWithPermission(AttendanceRequest request) {
        String employeeId = request.getEmployeeId();
        String timeStr = request.getTime();

        System.out.println("=== CHECKOUT WITH PERMISSION DEBUG ===");
        System.out.println("Employee ID: " + employeeId);
        System.out.println("Time: " + timeStr);

        if (employeeId == null || timeStr == null) {
            return "Missing employeeId or time";
        }

        LocalDate today = LocalDate.now();

        // Find attendance record
        List<Attendance> attendanceList = attendanceRepository.findByEmployeeIdAndDate(employeeId, today);

        if (attendanceList == null || attendanceList.isEmpty()) {
            return "No check-in found for today. Please check-in first before attempting checkout.";
        }

        Attendance attendance = attendanceList.get(0);

        // Validate existing checks (same as before)
        if (attendance.getCheckinTime() == null) {
            return "Check-in time not found for today. Please complete check-in first.";
        }

        if (attendance.getStatus() != null && attendance.getStatus().equals("Absent")) {
            return "Cannot checkout: Employee is marked as 'Absent' for today. Contact admin if this is incorrect.";
        }

        if (attendance.getCheckoutTime() != null) {
            return String.format("Already checked out today at %s. Multiple checkouts are not allowed.",
                    attendance.getCheckoutTime().format(DateTimeFormatter.ofPattern("HH:mm:ss")));
        }

        try {
            LocalTime checkoutTime = parseCheckoutTime(timeStr);

            if (checkoutTime.isBefore(attendance.getCheckinTime())) {
                return String.format("Checkout time (%s) cannot be before check-in time (%s)",
                        checkoutTime.format(DateTimeFormatter.ofPattern("HH:mm:ss")),
                        attendance.getCheckinTime().format(DateTimeFormatter.ofPattern("HH:mm:ss")));
            }

            // Set checkout time
            attendance.setCheckoutTime(checkoutTime);

            String finalStatus;
            String message;

            // Check if this was a permission day
            if ("Permission".equals(attendance.getStatus()) &&
                    attendance.getPermissionFrom() != null && attendance.getPermissionTo() != null) {

                // Process permission checkout
                Map<String, Object> permissionResult = processPermissionCheckout(
                        employeeId, checkoutTime, today,
                        attendance.getPermissionFrom(), attendance.getPermissionTo()
                );

                finalStatus = (String) permissionResult.get("finalStatus");
                message = (String) permissionResult.get("message");

            } else {
                // Normal checkout processing
                Map<String, LocalTime> workTiming = getEmployeeWorkTiming(employeeId);
                finalStatus = determineCheckoutStatusBasedOnDuration(employeeId, attendance.getCheckinTime(), checkoutTime, attendance.getStatus());
                String workDuration = calculateWorkDurationSummary(attendance.getCheckinTime(), checkoutTime, workTiming);
                message = String.format("Successfully checked out at %s. %s",
                        checkoutTime.format(DateTimeFormatter.ofPattern("HH:mm:ss")), workDuration);
            }

            attendance.setStatus(finalStatus);

            Attendance savedAttendance = attendanceRepository.save(attendance);
            System.out.println("Successfully processed checkout. Record ID: " + savedAttendance.getId() + ", Final Status: " + finalStatus);

            return String.format("Successfully checked out at %s. %s Status: %s",
                    checkoutTime.format(DateTimeFormatter.ofPattern("HH:mm:ss")),
                    message,
                    finalStatus);

        } catch (Exception e) {
            System.err.println("Error processing checkout: " + e.getMessage());
            e.printStackTrace();
            return "Error processing checkout: " + e.getMessage();
        }
    }
    /**
     * 21/6/25
     * ✅ NEW: Helper method to parse checkout time
     */
    private LocalTime parseCheckoutTime(String timeStr) {
        try {
            // Try parsing as ISO_INSTANT first
            return Instant.parse(timeStr)
                    .atZone(ZoneId.systemDefault())
                    .toLocalTime();
        } catch (Exception e1) {
            try {
                // Try parsing as plain time string
                return LocalTime.parse(timeStr);
            } catch (Exception e2) {
                throw new IllegalArgumentException("Invalid time format: " + timeStr);
            }
        }
    }

    /**
     * 21/6/25
     * ✅ NEW: Get current permission status for employee in given month
     */
    public PermissionStatusDTO getPermissionStatus(String employeeId, String yearMonth) {
        try {
            System.out.println("=== GETTING PERMISSION STATUS FOR EMPLOYEE ===");
            System.out.println("Employee ID: " + employeeId);
            System.out.println("Year-Month: " + yearMonth);

            // Parse year-month
            YearMonth ym = YearMonth.parse(yearMonth);
            LocalDate startDate = ym.atDay(1);
            LocalDate endDate = ym.atEndOfMonth();

            // Get all permission records from attendance table for this month
            List<Attendance> permissionRecords = attendanceRepository
                    .findByEmployeeIdAndDateBetweenAndStatus(employeeId, startDate, endDate, "Permission");

            System.out.println("Found " + permissionRecords.size() + " permission records in attendance table");

            // Calculate actual usage from attendance records
            int actualUsedMinutes = 0;
            int actualPermissionCount = 0;

            for (Attendance record : permissionRecords) {
                if (record.getPermissionFrom() != null && record.getPermissionTo() != null) {
                    long minutes = java.time.Duration.between(
                            record.getPermissionFrom(),
                            record.getPermissionTo()
                    ).toMinutes();
                    actualUsedMinutes += (int) minutes;
                    actualPermissionCount++;

                    System.out.println("Permission record: " + record.getDate() +
                            " from " + record.getPermissionFrom() +
                            " to " + record.getPermissionTo() +
                            " = " + minutes + " minutes");
                }
            }

            System.out.println("Calculated from attendance records:");
            System.out.println("  Total minutes used: " + actualUsedMinutes);
            System.out.println("  Permission count: " + actualPermissionCount);

            // Get or create permission tracker
            MonthlyPermissionTracker tracker = getOrCreatePermissionTracker(employeeId, yearMonth);

            // Check if tracker is out of sync with actual records
            if (tracker.getTotalPermissionMinutesUsed() != actualUsedMinutes ||
                    tracker.getPermissionCount() != actualPermissionCount) {

                System.out.println("SYNC ISSUE DETECTED!");
                System.out.println("Tracker shows: " + tracker.getTotalPermissionMinutesUsed() + " minutes, " + tracker.getPermissionCount() + " permissions");
                System.out.println("Actual records: " + actualUsedMinutes + " minutes, " + actualPermissionCount + " permissions");

                // Sync tracker with actual records
                tracker.setTotalPermissionMinutesUsed(actualUsedMinutes);
                tracker.setPermissionCount(actualPermissionCount);

                // Update grace usage based on actual records
                if (actualPermissionCount > 0) {
                    tracker.setFirstPermissionGraceUsed(true);
                }
                if (actualPermissionCount > 1) {
                    tracker.setSecondPermissionGraceUsed(true);
                }

                // Save synchronized tracker
                permissionTrackerRepository.save(tracker);
                System.out.println("Tracker synchronized with actual attendance records");
            }

            // Use actual values for calculations
            int usedMinutes = actualUsedMinutes;
            int remainingMinutes = Math.max(0, 120 - usedMinutes); // 2 hours = 120 minutes
            int permissionCount = actualPermissionCount;
            int remainingPermissions = Math.max(0, 2 - permissionCount);

            boolean canRequestPermission = remainingMinutes > 0 && remainingPermissions > 0;
            String message;

            if (usedMinutes >= 120) {
                message = "Permission limit exceeded. Monthly 2-hour permission limit has been exhausted.";
                canRequestPermission = false;
            } else if (permissionCount >= 2) {
                message = "Maximum permissions used. Only 2 permissions allowed per month.";
                canRequestPermission = false;
            } else if (remainingMinutes < 60) {
                message = String.format("Insufficient time remaining. Only %d minutes left (minimum 60 minutes required).", remainingMinutes);
                canRequestPermission = false;
            } else {
                message = String.format("Permission available. %d minutes remaining, %d permissions left.", remainingMinutes, remainingPermissions);
            }

            System.out.println("Final Permission Status Summary:");
            System.out.println("  Used Minutes: " + usedMinutes);
            System.out.println("  Remaining Minutes: " + remainingMinutes);
            System.out.println("  Permission Count: " + permissionCount);
            System.out.println("  Can Request: " + canRequestPermission);
            System.out.println("  Message: " + message);

            return new PermissionStatusDTO(usedMinutes, remainingMinutes, permissionCount,
                    remainingPermissions, canRequestPermission, message);

        } catch (Exception e) {
            System.err.println("Error getting permission status: " + e.getMessage());
            e.printStackTrace();

            return new PermissionStatusDTO(0, 120, 0, 2, false,
                    "Error retrieving permission status: " + e.getMessage());
        }
    }

    /**
     * ✅ UPDATED: Enhanced updateStatus method with permission validation
     */
    public ResponseEntity<Map<String, Object>> updateStatus(@RequestBody Map<String, Object> request) {
        try {
            System.out.println("=== UPDATE STATUS REQUEST ===");
            System.out.println("Request data: " + request);

            String employeeId = (String) request.get("employeeId");
            String employeeName = (String) request.get("employeeName");
            String dateStr = (String) request.get("date");
            String status = (String) request.get("status");
            String reason = (String) request.get("reason");

            // Validate required fields
            if (employeeId == null || employeeName == null || dateStr == null || status == null || reason == null) {
                return ResponseEntity.badRequest().body(Map.of("error", "Missing required fields"));
            }

            LocalDate date = LocalDate.parse(dateStr);

            // Special validation for Permission status
            if ("Permission".equals(status)) {
                String permissionFromStr = (String) request.get("permissionFrom");
                String permissionToStr = (String) request.get("permissionTo");

                if (permissionFromStr == null || permissionToStr == null) {
                    return ResponseEntity.badRequest().body(Map.of("error", "Permission times are required for Permission status"));
                }

                try {
                    LocalTime permissionFrom = LocalTime.parse(permissionFromStr);
                    LocalTime permissionTo = LocalTime.parse(permissionToStr);

                    // Validate permission request against monthly limits
                    Map<String, Object> validationResult = this.validatePermissionRequest(
                            employeeId, permissionFrom, permissionTo, date);

                    if (!(Boolean) validationResult.get("valid")) {
                        String errorReason = (String) validationResult.get("reason");
                        String errorMessage = (String) validationResult.get("message");

                        Map<String, Object> errorResponse = new HashMap<>();
                        errorResponse.put("error", errorMessage);
                        errorResponse.put("reason", errorReason);

                        // Add additional context for UI
                        if ("MONTHLY_LIMIT_EXHAUSTED".equals(errorReason) || "PERMISSION_COUNT_EXCEEDED".equals(errorReason)) {
                            errorResponse.put("permissionExhausted", true);
                        } else if ("EXCEEDS_REMAINING_TIME".equals(errorReason)) {
                            errorResponse.put("exceedsLimit", true);
                            if (validationResult.containsKey("currentRemainingMinutes")) {
                                errorResponse.put("remainingMinutes", validationResult.get("currentRemainingMinutes"));
                            }
                        }

                        return ResponseEntity.badRequest().body(errorResponse);
                    }

                    // Add permission times to request for processing
                    request.put("permissionFrom", permissionFromStr);
                    request.put("permissionTo", permissionToStr);

                } catch (Exception e) {
                    return ResponseEntity.badRequest().body(Map.of("error", "Invalid permission time format"));
                }
            }

            // Proceed with status update
            Map<String, Object> result = this.updateEmployeeStatus(request);

            if (result.containsKey("error")) {
                return ResponseEntity.badRequest().body(result);
            }

            return ResponseEntity.ok(result);

        } catch (Exception e) {
            System.err.println("Error updating status: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Error updating status: " + e.getMessage()));
        }
    }
    /**
     * ✅ UPDATED: Enhanced updateEmployeeStatus method in AttendanceService
     */
    public Map<String, Object> updateEmployeeStatus(Map<String, Object> request) {
        Map<String, Object> result = new HashMap<>();

        try {
            String employeeId = (String) request.get("employeeId");
            String employeeName = (String) request.get("employeeName");
            String dateStr = (String) request.get("date");
            String status = (String) request.get("status");
            String reason = (String) request.get("reason");

            LocalDate date = LocalDate.parse(dateStr);

            System.out.println("=== UPDATING EMPLOYEE STATUS ===");
            System.out.println("Employee ID: " + employeeId);
            System.out.println("Date: " + date);
            System.out.println("Status: " + status);

            // Check if attendance record exists
            List<Attendance> existingRecords = attendanceRepository.findByEmployeeIdAndDate(employeeId, date);
            Attendance attendance;

            if (existingRecords.isEmpty()) {
                // Create new attendance record
                attendance = new Attendance();
                attendance.setEmployeeId(employeeId);
                attendance.setEmployeeName(employeeName);
                attendance.setDate(date);

                // Set department and designation from employee record
                Optional<Employee> employeeOpt = employeeRepository.findById(employeeId);
                if (employeeOpt.isPresent()) {
                    Employee emp = employeeOpt.get();
                    attendance.setDepartment(emp.getDepartment());
                    attendance.setDesignation(emp.getDesignation());
                }
            } else {
                attendance = existingRecords.get(0);
            }

            // Update basic fields
            attendance.setStatus(status);
            attendance.setReason(reason);

            // Handle Permission status with monthly tracking
            if ("Permission".equals(status)) {
                String permissionFromStr = (String) request.get("permissionFrom");
                String permissionToStr = (String) request.get("permissionTo");

                if (permissionFromStr != null && permissionToStr != null) {
                    LocalTime permissionFrom = LocalTime.parse(permissionFromStr);
                    LocalTime permissionTo = LocalTime.parse(permissionToStr);

                    attendance.setPermissionFrom(permissionFrom);
                    attendance.setPermissionTo(permissionTo);

                    // Update monthly permission tracker
                    String yearMonth = YearMonth.from(date).toString();
                    MonthlyPermissionTracker tracker = getOrCreatePermissionTracker(employeeId, yearMonth);

                    long permissionMinutes = java.time.Duration.between(permissionFrom, permissionTo).toMinutes();

                    // Update tracker
                    tracker.setTotalPermissionMinutesUsed(tracker.getTotalPermissionMinutesUsed() + (int) permissionMinutes);
                    tracker.setPermissionCount(tracker.getPermissionCount() + 1);

                    permissionTrackerRepository.save(tracker);

                    System.out.println("Updated permission tracker:");
                    System.out.println("  Total minutes used: " + tracker.getTotalPermissionMinutesUsed());
                    System.out.println("  Permission count: " + tracker.getPermissionCount());
                }
            }

            // Handle Half Day Leave status
            if ("Half Day Leave".equals(status)) {
                String halfDayType = (String) request.get("halfDayType");
                String halfDayFromStr = (String) request.get("halfDayFrom");
                String halfDayToStr = (String) request.get("halfDayTo");

                if (halfDayType != null) {
                    attendance.setHalfdayType(halfDayType);
                }
                if (halfDayFromStr != null && halfDayToStr != null) {
                    attendance.setHalfdayFrom(LocalTime.parse(halfDayFromStr));
                    attendance.setHalfdayTo(LocalTime.parse(halfDayToStr));
                }
            }

            // Save attendance record
            Attendance savedAttendance = attendanceRepository.save(attendance);

            result.put("success", true);
            result.put("message", String.format("Status updated successfully for %s on %s", employeeName, date));
            result.put("attendanceId", savedAttendance.getId());
            result.put("status", status);

            // Add permission details to response if applicable
            if ("Permission".equals(status)) {
                String yearMonth = YearMonth.from(date).toString();
                PermissionStatusDTO statusDto = getPermissionStatus(employeeId, yearMonth);
                result.put("permissionStatus", statusDto);
            }

            return result;

        } catch (Exception e) {
            System.err.println("Error updating employee status: " + e.getMessage());
            e.printStackTrace();

            result.put("error", "Error updating status: " + e.getMessage());
            return result;
        }
    }
//    /**
//     * Find attendance records by employee ID and date
//     */
//    public List<Attendance> findByEmployeeIdAndDate(String employeeId, LocalDate date) {
//        return attendanceRepository.findByEmployeeIdAndDate(employeeId, date);
//    }
//    /**
//     * Get employee by ID
//     */
//    public Employee getEmployeeById(String employeeId) {
//        return employeeRepository.findById(employeeId).orElse(null);
//    }
//    /**
//     * Validate if face recognition can be used for employee
//     */
//    public Map<String, Object> validateFaceRecognitionAccess(String employeeId) {
//        Map<String, Object> result = new HashMap<>();
//
//        try {
//            LocalDate today = LocalDate.now();
//            LocalTime currentTime = LocalTime.now();
//
//            // Get employee
//            Employee employee = getEmployeeById(employeeId);
//            if (employee == null) {
//                result.put("valid", false);
//                result.put("message", "Employee not found");
//                return result;
//            }
//
//            // Get attendance record
//            List<Attendance> records = findByEmployeeIdAndDate(employeeId, today);
//            if (records.isEmpty()) {
//                result.put("valid", false);
//                result.put("message", "No permission allocation found for today");
//                return result;
//            }
//
//            Attendance attendance = records.get(0);
//
//            // Check permission status
//            if (!"Permission".equals(attendance.getStatus()) ||
//                    attendance.getPermissionFrom() == null ||
//                    attendance.getPermissionTo() == null) {
//                result.put("valid", false);
//                result.put("message", "No valid permission found for today");
//                return result;
//            }
//
//            // Check time window
//            boolean withinTime = !currentTime.isBefore(attendance.getPermissionFrom()) &&
//                    !currentTime.isAfter(attendance.getPermissionTo().plusMinutes(5));
//
//            if (!withinTime) {
//                result.put("valid", false);
//                result.put("message", String.format("Outside permission time. Allowed: %s to %s",
//                        attendance.getPermissionFrom().format(DateTimeFormatter.ofPattern("HH:mm")),
//                        attendance.getPermissionTo().format(DateTimeFormatter.ofPattern("HH:mm"))));
//                return result;
//            }
//
//            result.put("valid", true);
//            result.put("message", "Face recognition allowed");
//            result.put("attendance", attendance);
//            result.put("employee", employee);
//
//            return result;
//
//        } catch (Exception e) {
//            System.err.println("Error validating face recognition access: " + e.getMessage());
//            result.put("valid", false);
//            result.put("message", "Error validating access: " + e.getMessage());
//            return result;
//        }
//    }

    /// 23/6/23

    public ResponseEntity<?> processPermission(AttendanceRequest request) {
        LocalDate today = LocalDate.now();
        Optional<Attendance> optionalAttendance = attendanceRepository
                .findAttendanceByEmployeeIdAndDate(request.getEmployeeId(), today);

        Attendance attendance = optionalAttendance.orElseGet(() -> {
            Attendance newEntry = new Attendance();
            newEntry.setEmployeeId(request.getEmployeeId());
            newEntry.setEmployeeName(request.getEmployeeName());
            newEntry.setDepartment(request.getDepartment());
            newEntry.setDesignation(request.getDesignation());
            newEntry.setDate(today);
            return newEntry;
        });

        LocalTime currentTime;
        try {
            currentTime = request.getTimeAsLocalTime();
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Invalid time format in request");
        }

        // Only update permissionFrom and permissionTo if provided
        if (request.getPermissionFrom() != null) {
            attendance.setPermissionFrom(request.getPermissionFrom());
        }
        if (request.getPermissionTo() != null) {
            attendance.setPermissionTo(request.getPermissionTo());
        }

        attendance.setStatus("Permission");

        LocalTime savedFrom = attendance.getPermissionFrom();
        LocalTime savedTo = attendance.getPermissionTo();

        if (attendance.getCheckinTime() == null &&
                savedFrom != null && savedTo != null &&
                currentTime.isAfter(savedFrom) &&
                currentTime.isBefore(savedTo)) {
            attendance.setCheckinTime(currentTime);
        } else if (attendance.getCheckinTime() != null) {
            return ResponseEntity.ok().body("Already Checked-in");
        }

        // Handle checkout time based on gender and permissionFrom
        if (savedFrom != null) {
            if ("male".equalsIgnoreCase(request.getGender())) {
                if (savedFrom.equals(LocalTime.of(17, 0)) || savedFrom.equals(LocalTime.of(16, 0))) {
                    attendance.setCheckoutTime(LocalTime.of(18, 0));
                }
            } else {
                if (savedFrom.equals(LocalTime.of(16, 0)) || savedFrom.equals(LocalTime.of(15, 0))) {
                    attendance.setCheckoutTime(LocalTime.of(17, 0));
                }
            }
        }

        attendanceRepository.save(attendance);
        return ResponseEntity.ok().body("Permission processed successfully");
    }

    public ResponseEntity<?> processHalfDay(AttendanceRequest request) {
        LocalDate today = LocalDate.now();
        Optional<Attendance> optionalAttendance = attendanceRepository
                .findAttendanceByEmployeeIdAndDate(request.getEmployeeId(), today);

        Attendance attendance = optionalAttendance.orElseGet(() -> {
            Attendance newEntry = new Attendance();
            newEntry.setEmployeeId(request.getEmployeeId());
            newEntry.setEmployeeName(request.getEmployeeName());
            newEntry.setDepartment(request.getDepartment());
            newEntry.setDesignation(request.getDesignation());
            newEntry.setDate(today);
            return newEntry;
        });

        // Safely update halfdayType only if it's present in request
        if (request.getHalfdayType() != null) {
            attendance.setHalfdayType(request.getHalfdayType());
        }

        attendance.setStatus("Half Day");

        LocalTime currentTime;
        try {
            currentTime = request.getTimeAsLocalTime();
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Invalid time format in request");
        }

        String halfType = attendance.getHalfdayType(); // use saved type
        if ("first".equalsIgnoreCase(halfType)) {
            attendance.setHalfdayFrom(LocalTime.of(9, 30));
            attendance.setHalfdayTo(LocalTime.of(14, 30));

            if (attendance.getCheckinTime() == null &&
                    currentTime != null &&
                    currentTime.isBefore(LocalTime.of(14, 30))) {
                attendance.setCheckinTime(currentTime);
            }

        } else if ("second".equalsIgnoreCase(halfType)) {
            attendance.setHalfdayFrom(LocalTime.of(13, 30));

            if ("male".equalsIgnoreCase(request.getGender())) {
                attendance.setHalfdayTo(LocalTime.of(18, 0));
                attendance.setCheckoutTime(LocalTime.of(18, 0));
            } else {
                attendance.setHalfdayTo(LocalTime.of(17, 0));
                attendance.setCheckoutTime(LocalTime.of(17, 0));
            }
        }

        attendanceRepository.save(attendance);
        return ResponseEntity.ok().body("Half Day processed successfully");
    }
    /// /------------------------------------------------------///
    /// //23/6/25

    public FaceRecognitionResult handlePermissionCheck(FaceRecognitionEntry entry) {
        String empId = entry.getEmployeeId();
        LocalDate date = entry.getTime().toLocalDate();
        LocalTime now = entry.getTime().toLocalTime();

        System.out.println("=== PERMISSION CHECK DEBUG ===");
        System.out.println("Employee ID: " + empId);
        System.out.println("Date: " + date);
        System.out.println("Current Time: " + now);

        // ✅ FIXED: Use the new method that returns Optional<Attendance>
        Optional<Attendance> allocatedOpt = attendanceRepository.findPermissionAllocation(empId, date);

        if (allocatedOpt.isEmpty()) {
            System.out.println("No permission allocation found for today");
            return new FaceRecognitionResult("❌ No permission allocated for today", "FAILED");
        }

        Attendance allocated = allocatedOpt.get();
        LocalTime permissionFrom = allocated.getPermissionFrom();
        LocalTime permissionTo = allocated.getPermissionTo();

        System.out.println("Permission From: " + permissionFrom);
        System.out.println("Permission To: " + permissionTo);

        // Check if employee comes within allocated permission time
        if (now.isAfter(permissionFrom) && now.isBefore(permissionTo)) {
            // Employee came within permission time - keep status as PERMISSION
            System.out.println("Employee came within permission time");

            allocated.setCheckinTime(now);
            allocated.setStatus("PERMISSION"); // Keep same status
            attendanceRepository.save(allocated);

            return new FaceRecognitionResult("✅ You came within permission time! Status remains: Permission", "PERMISSION");
        } else {
            // Employee came outside permission time - mark as LATE
            System.out.println("Employee came outside permission time");

            allocated.setCheckinTime(now);
            allocated.setStatus("LATE");
            attendanceRepository.save(allocated);

            String message = String.format("❌ You came late! Expected: %s-%s, Actual: %s",
                    permissionFrom, permissionTo, now);
            return new FaceRecognitionResult(message, "LATE");
        }
    }

    public FaceRecognitionResult handleHalfDayCheck(FaceRecognitionEntry entry) {
        String empId = entry.getEmployeeId();
        LocalDate date = entry.getTime().toLocalDate();
        LocalTime now = entry.getTime().toLocalTime();

        System.out.println("=== HALF DAY CHECK DEBUG ===");
        System.out.println("Employee ID: " + empId);
        System.out.println("Date: " + date);
        System.out.println("Current Time: " + now);

        // ✅ FIXED: Use the new method that returns Optional<Attendance>
        Optional<Attendance> allocatedOpt = attendanceRepository.findHalfDayAllocation(empId, date);

        if (allocatedOpt.isEmpty()) {
            System.out.println("No half day allocation found for today");
            return new FaceRecognitionResult("❌ No half day allocated for today", "FAILED");
        }

        Attendance allocated = allocatedOpt.get();
        LocalTime halfdayFrom = allocated.getHalfdayFrom();
        LocalTime halfdayTo = allocated.getHalfdayTo();

        System.out.println("Half Day From: " + halfdayFrom);
        System.out.println("Half Day To: " + halfdayTo);

        // Check if employee comes within allocated half day time
        if (now.isAfter(halfdayFrom) && now.isBefore(halfdayTo)) {
            // Employee came within half day time - keep status as HALF_DAY
            System.out.println("Employee came within half day time");

            allocated.setCheckinTime(now);
            allocated.setStatus("HALF_DAY"); // Keep same status
            attendanceRepository.save(allocated);

            return new FaceRecognitionResult("✅ You came within half day time! Status remains: Half Day", "HALF_DAY");
        } else {
            // Employee came outside half day time - mark as ABSENT
            System.out.println("Employee came outside half day time");

            allocated.setCheckinTime(now);
            allocated.setStatus("ABSENT");
            attendanceRepository.save(allocated);

            String message = String.format("❌ You missed the half day time! Expected: %s-%s, Actual: %s",
                    halfdayFrom, halfdayTo, now);
            return new FaceRecognitionResult(message, "ABSENT");
        }
    }

    // ✅ NEW: Quick check methods for admin dashboard
    public boolean hasPermissionToday(String employeeId) {
        return attendanceRepository.hasPermissionAllocated(employeeId, LocalDate.now());
    }

    public boolean hasHalfDayToday(String employeeId) {
        return attendanceRepository.hasHalfDayAllocated(employeeId, LocalDate.now());
    }

    // ✅ NEW: Get all permission/halfday status for admin
    public List<Attendance> getTodayPermissionHalfdayStatus() {
        return attendanceRepository.findAllPermissionHalfdayAllocations(LocalDate.now());
    }

    // ✅ NEW: Get employees who haven't used their allocation yet
    public List<Attendance> getUnusedAllocations() {
        return attendanceRepository.findUnusedPermissionHalfdayAllocations(LocalDate.now());
    }

    // ✅ NEW: Get employees who came late for their allocation
    public List<Attendance> getLateAllocationUsers() {
        return attendanceRepository.findLatePermissionHalfdayUsers(LocalDate.now());
    }

//29/6

    @Transactional
    public HolidayMarkingResult markHolidayForAllEmployees(LocalDate date, String reason, Boolean isAdminOverride) {
        HolidayMarkingResult result = new HolidayMarkingResult();

        try {
            List<Employee> allEmployees = attendanceRepository.findAllEmployees();

            int employeesMarked = 0;
            int existingRecordsUpdated = 0;

            for (Employee employee : allEmployees) {
                Optional<Attendance> existingAttendance = attendanceRepository
                        .findAttendanceByEmployeeIdAndDate(employee.getId(), date);

                if (existingAttendance.isPresent()) {
                    if (isAdminOverride || canUpdateStatus(existingAttendance.get().getStatus())) {
                        Attendance attendance = existingAttendance.get();
                        attendance.setStatus("HOLIDAY");
                        attendance.setReason(reason);
                        attendanceRepository.save(attendance);
                        existingRecordsUpdated++;
                    }
                } else {
                    Attendance newAttendance = new Attendance();
                    newAttendance.setEmployeeId(employee.getId());
                    newAttendance.setEmployeeName(employee.getUsername());
                    newAttendance.setDepartment(employee.getDepartment());
                    newAttendance.setDesignation(employee.getDesignation());
                    newAttendance.setDate(date);
                    newAttendance.setStatus("HOLIDAY");
                    newAttendance.setReason(reason);
                    attendanceRepository.save(newAttendance);
                    employeesMarked++;
                }
            }

            result.setEmployeesMarked(employeesMarked);
            result.setExistingRecordsUpdated(existingRecordsUpdated);
            result.setMessage("Holiday marked successfully for " + (employeesMarked + existingRecordsUpdated) + " employees");

        } catch (Exception e) {
            result.setMessage("Error marking holiday: " + e.getMessage());
            result.setEmployeesMarked(0);
            result.setExistingRecordsUpdated(0);
        }

        return result;
    }

    private boolean canUpdateStatus(String currentStatus) {
        return currentStatus == null ||
                (!currentStatus.equals("PRESENT") && !currentStatus.equals("ABSENT"));
    }

}


