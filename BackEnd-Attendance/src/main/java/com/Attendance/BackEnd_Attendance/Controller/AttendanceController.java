package com.Attendance.BackEnd_Attendance.Controller;

import com.Attendance.BackEnd_Attendance.Model.*;
import com.Attendance.BackEnd_Attendance.Repository.AttendanceRepository;
import com.Attendance.BackEnd_Attendance.Repository.EmployeeRepository;
import com.Attendance.BackEnd_Attendance.Service.AttendanceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/attendance")
@CrossOrigin(origins = "*")
public class AttendanceController {

    @Autowired
    private AttendanceService attendanceService;

    @Autowired
    private AttendanceRepository attendanceRepository;

    @Autowired
    private EmployeeRepository employeeRepository;

    // Helper method to convert "On Leave" status to "Absent" for display
    private String getDisplayStatus(String originalStatus) {
        if ("On Leave".equals(originalStatus)) {
            return "Absent";
        }
        return originalStatus;
    }

    // Check if attendance is already marked for a specific date
    @GetMapping("/check-status/{employeeId}/{date}")
    public ResponseEntity<Map<String, Object>> checkAttendanceStatus(
            @PathVariable String employeeId,
            @PathVariable String date) {

        Map<String, Object> response = new HashMap<>();

        try {
            LocalDate checkDate = LocalDate.parse(date, DateTimeFormatter.ISO_LOCAL_DATE);
            List<Attendance> attendanceRecords = attendanceRepository.findByEmployeeIdAndDate(employeeId, checkDate);

            if (attendanceRecords.isEmpty()) {
                response.put("isMarked", false);
                response.put("message", "No attendance record found for " + date);
                response.put("employeeId", employeeId);
                response.put("date", date);
                return ResponseEntity.ok(response);
            } else {
                Attendance attendance = attendanceRecords.get(0);
                String displayStatus = getDisplayStatus(attendance.getStatus());

                response.put("isMarked", true);
                response.put("message", "Attendance already marked for " + date);
                response.put("employeeId", employeeId);
                response.put("employeeName", attendance.getEmployeeName());
                response.put("department", attendance.getDepartment());
                response.put("date", date);
                response.put("status", displayStatus); // Changed from "currentStatus" to "status" to match frontend
                response.put("checkinTime", attendance.getCheckinTime() != null ?
                        attendance.getCheckinTime().toString() : null);
                response.put("checkoutTime", attendance.getCheckoutTime() != null ?
                        attendance.getCheckoutTime().toString() : null);
                return ResponseEntity.ok(response);
            }
        } catch (Exception e) {
            response.put("isMarked", false);
            response.put("message", "Error checking attendance status: " + e.getMessage());
            response.put("error", true);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }

    @PostMapping("/checkin")
    public ResponseEntity<AttendanceResponse> checkin(@RequestBody Map<String, String> requestBody) {
        try {
            String employeeId = requestBody.get("employeeId");

            // Let the service handle all the business logic including existing record checks
            Attendance attendance = attendanceService.checkIn(requestBody);

            AttendanceResponse response = new AttendanceResponse();
            response.setStatus("success");

            // Determine the appropriate message based on the attendance status
            String message = "Check-in successful";
            if ("Permission".equals(attendance.getStatus())) {
                message = "Check-in successful - Permission status maintained";
            } else if ("Half Day Leave".equals(attendance.getStatus())) {
                message = "Check-in successful - Half Day Leave status maintained";
            } else if ("Late".equals(attendance.getStatus())) {
                message = "Check-in successful - Marked as Late";
            }

            response.setMessage(message);
            response.setEmployeeName(attendance.getEmployeeName());
            response.setAlreadyMarked(false);
            response.setNewCheckinTime(attendance.getCheckinTime() != null ?
                    attendance.getCheckinTime().toString() : null);
            response.setNewStatus(getDisplayStatus(attendance.getStatus()));

            return ResponseEntity.ok(response);

        } catch (IllegalArgumentException e) {
            // Handle the case where attendance is already marked (thrown by service)
            if (e.getMessage().contains("Attendance already registered")) {
                // Parse existing attendance info from the exception message if needed
                AttendanceResponse errorResponse = new AttendanceResponse();
                errorResponse.setStatus("error");
                errorResponse.setMessage(e.getMessage());
                errorResponse.setEmployeeName(null);
                errorResponse.setAlreadyMarked(true);
                return ResponseEntity.status(HttpStatus.CONFLICT).body(errorResponse);
            }

            AttendanceResponse errorResponse = new AttendanceResponse();
            errorResponse.setStatus("error");
            errorResponse.setMessage(e.getMessage());
            errorResponse.setEmployeeName(null);
            errorResponse.setAlreadyMarked(false);
            return ResponseEntity.badRequest().body(errorResponse);
        } catch (Exception e) {
            AttendanceResponse errorResponse = new AttendanceResponse();
            errorResponse.setStatus("error");
            errorResponse.setMessage("An unexpected error occurred: " + e.getMessage());
            errorResponse.setEmployeeName(null);
            errorResponse.setAlreadyMarked(false);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @GetMapping("/validate-checkout/{employeeId}")
    public ResponseEntity<Map<String, Object>> validateCheckoutEligibility(@PathVariable String employeeId) {
        try {
            Map<String, Object> validation = attendanceService.validateCheckoutEligibility(employeeId);

            // Update status for display if needed
            if (validation.containsKey("currentStatus")) {
                String displayStatus = getDisplayStatus((String) validation.get("currentStatus"));
                validation.put("currentStatus", displayStatus);
            }

            if ((Boolean) validation.get("eligible")) {
                return ResponseEntity.ok(validation);
            } else {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(validation);
            }
        } catch (Exception e) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("eligible", false);
            errorResponse.put("reason", "System error");
            errorResponse.put("message", "Error validating checkout: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @PostMapping("/checkout")
    public ResponseEntity<?> checkout(@RequestBody AttendanceRequest request) {
        try {
            Map<String, Object> validation = attendanceService.validateCheckoutEligibility(request.getEmployeeId());

            if (!(Boolean) validation.get("eligible")) {
                Map<String, String> response = new HashMap<>();
                response.put("message", (String) validation.get("message"));
                response.put("reason", (String) validation.get("reason"));
                response.put("status", "error");
                return ResponseEntity.badRequest().body(response);
            }

            String result = attendanceService.recordCheckout(request);
            Map<String, String> response = new HashMap<>();

            if (result.contains("No check-in") ||
                    result.contains("Cannot checkout") ||
                    result.contains("Already checked out") ||
                    result.contains("Check-in time not found") ||
                    result.contains("cannot be before") ||
                    result.contains("cannot be in the future")) {

                response.put("message", result);
                response.put("status", "error");
                return ResponseEntity.badRequest().body(response);
            }

            response.put("message", result);
            response.put("status", "success");

            if (result.contains("Successfully checked out")) {
                response.put("checkoutCompleted", "true");
                if (result.contains("at ")) {
                    String[] parts = result.split("at ");
                    if (parts.length > 1) {
                        String timeInfo = parts[1].split("\\.")[0];
                        response.put("checkoutTime", timeInfo);
                    }
                }
            }

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Check-out failed: " + e.getMessage());
            errorResponse.put("status", "error");
            errorResponse.put("error", "system_error");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }


    @GetMapping("/current-status/{employeeId}")
    public ResponseEntity<Map<String, Object>> getCurrentAttendanceStatus(@PathVariable String employeeId) {
        try {
            LocalDate today = LocalDate.now();
            Map<String, Object> statusInfo = attendanceService.checkAttendanceStatus(employeeId, today);

            // Update status for display if needed
            if (statusInfo.containsKey("currentStatus")) {
                String displayStatus = getDisplayStatus((String) statusInfo.get("currentStatus"));
                statusInfo.put("currentStatus", displayStatus);
            }

            return ResponseEntity.ok(statusInfo);
        } catch (Exception e) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("error", true);
            errorResponse.put("message", "Error fetching current status: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @GetMapping("/summary/{employeeId}")
    public ResponseEntity<?> getSummary(@PathVariable String employeeId) {
        try {
            AttendanceSummary summary = attendanceService.getAttendanceSummary(employeeId);
            // Update today's status for display
            summary.setTodayStatus(getDisplayStatus(summary.getTodayStatus()));
            return ResponseEntity.ok(summary);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error: " + e.getMessage());
        }
    }

    @GetMapping("/dashboard-stats")
    public ResponseEntity<AdminDashboardStats> getDashboardStats() {
        AdminDashboardStats stats = attendanceService.getAdminDashboardStats();
        return ResponseEntity.ok(stats);
    }

    @GetMapping("/today-status")
    public ResponseEntity<List<EmployeeStatus>> getTodayEmployeeStatus() {
        List<EmployeeStatus> statusList = attendanceService.getAllEmployeeStatusToday();

        // Update status for display - convert "On Leave" to "Absent"
        for (EmployeeStatus status : statusList) {
            status.setStatus(getDisplayStatus(status.getStatus()));
        }

        return ResponseEntity.ok(statusList);
    }

    @GetMapping("/today-attendance-status")
    public List<EmployeeAttendanceDTO> getTodayStatus() {
        return attendanceService.getTodayEmployeeAttendance();
    }

    @GetMapping("/report")
    public List<EmployeeAttendanceDTO> getAttendanceReport() {
        return attendanceService.getAttendanceReport();
    }

    // Updated getEmployeeById method to include gender
    @GetMapping("/employee/{employeeId}")
    public ResponseEntity<?> getEmployeeById(@PathVariable String employeeId) {
        try {
            Optional<Employee> employee = employeeRepository.findById(employeeId);
            if (employee.isPresent()) {
                Employee emp = employee.get();
                Map<String, String> response = new HashMap<>();
                response.put("id", emp.getId());
                response.put("username", emp.getUsername());
                response.put("department", emp.getDepartment());
                response.put("designation", emp.getDesignation());
                response.put("email", emp.getEmail());
                response.put("usertype", emp.getUsertype());
                response.put("gender", emp.getGender()); // Add gender field
                return ResponseEntity.ok(response);
            } else {
                Map<String, String> errorResponse = new HashMap<>();
                errorResponse.put("message", "Employee not found");
                errorResponse.put("status", "error");
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(errorResponse);
            }
        } catch (Exception e) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Error fetching employee: " + e.getMessage());
            errorResponse.put("status", "error");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }
    // BACKEND - Updated updateStatus method in AttendanceController
    @PostMapping("/update-status")
    public ResponseEntity<Map<String, String>> updateStatus(@RequestBody AttendanceStatusRequest request) {
        Map<String, String> response = new HashMap<>();

        try {
            // Debug log - Log the incoming request
            System.out.println("=== INCOMING REQUEST ===");
            System.out.println("Request: " + request.toString());
            System.out.println("Employee Gender: " + request.getEmployeeGender());
            System.out.println("Mark Checkout Time: " + request.getMarkCheckoutTime());
            System.out.println("Checkout Time: " + request.getCheckoutTime());
            System.out.println("Is Admin Override: " + request.getIsAdminOverride());
            System.out.println("Has Existing Status: " + request.getHasExistingStatus());

            // Validate employee exists
            Optional<Employee> employeeOpt = employeeRepository.findById(request.getEmployeeId());
            if (!employeeOpt.isPresent()) {
                response.put("status", "error");
                response.put("message", "Employee ID '" + request.getEmployeeId() + "' not found in database");
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(response);
            }

            Employee employee = employeeOpt.get();

            // Validate employee name matches - NULL CHECK ADDED HERE
            String requestEmployeeName = request.getEmployeeName();
            if (requestEmployeeName == null || requestEmployeeName.trim().isEmpty()) {
                response.put("status", "error");
                response.put("message", "Employee name is required");
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            }

            if (!employee.getUsername().equalsIgnoreCase(requestEmployeeName.trim())) {
                response.put("status", "error");
                response.put("message", "Employee name mismatch. Expected: " + employee.getUsername() +
                        ", Provided: " + requestEmployeeName);
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            }

            // Check if status already exists for this date
            List<Attendance> existingRecords = attendanceRepository.findByEmployeeIdAndDate(
                    request.getEmployeeId(), request.getDate());

            String previousStatus = null;
            boolean statusOverridden = false;
            Attendance attendance = null;

            if (!existingRecords.isEmpty()) {
                // UPDATE EXISTING RECORD INSTEAD OF DELETING
                attendance = existingRecords.get(0);
                String currentStatus = attendance.getStatus();
                previousStatus = currentStatus;

                // If this is an admin override request, allow the update
                if (request.getIsAdminOverride() != null && request.getIsAdminOverride()) {
                    System.out.println("Admin override detected - updating existing record from " + currentStatus + " to " + request.getStatus());
                    statusOverridden = true;

                    // PRESERVE EXISTING CHECK-IN DATA
                    // Don't delete the record, just update the status and related fields
                    System.out.println("Preserving existing check-in data: " + attendance.getCheckinTime());
                } else {
                    // Original validation logic for non-admin requests
                    if ("Absent".equals(currentStatus) &&
                            ("Permission".equals(request.getStatus()) || "Half Day Leave".equals(request.getStatus()))) {

                        response.put("status", "error");
                        response.put("message", "Cannot update to " + request.getStatus() +
                                " because employee is marked as ABSENT on " + request.getDate() +
                                ". Please change the absent status first.");
                        return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
                    }

                    // If trying to update to different status without admin override, reject
                    if (!currentStatus.equals(request.getStatus())) {
                        response.put("status", "error");
                        response.put("message", "Status already exists for employee " + employee.getUsername() +
                                " on " + request.getDate() + ". Current status: " + currentStatus +
                                ". Only one status update per day is allowed unless updating the same status.");
                        return ResponseEntity.status(HttpStatus.CONFLICT).body(response);
                    }
                }
            } else {
                // Create new attendance record only if no existing record
                attendance = new Attendance();
                attendance.setEmployeeId(request.getEmployeeId());
                attendance.setDate(request.getDate());
                attendance.setEmployeeName(employee.getUsername());
                attendance.setDepartment(employee.getDepartment());
                attendance.setDesignation(employee.getDesignation());
            }

            // Update the status and reason
            String statusToStore = request.getStatus();
            attendance.setStatus(statusToStore);
            attendance.setReason(request.getReason());

            boolean checkoutMarked = false;
            String checkoutTimeStr = null;

            // Handle different status types
            switch (statusToStore) {
                case "Permission Granted":
                case "Permission":
                    if (request.getPermissionFrom() == null || request.getPermissionTo() == null) {
                        response.put("status", "error");
                        response.put("message", "Permission times are required for Permission status");
                        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
                    }

                    // Validate permission times
                    if (request.getPermissionFrom().isAfter(request.getPermissionTo()) ||
                            request.getPermissionFrom().equals(request.getPermissionTo())) {
                        response.put("status", "error");
                        response.put("message", "Permission 'To Time' must be after 'From Time'");
                        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
                    }

                    attendance.setPermissionFrom(request.getPermissionFrom());
                    attendance.setPermissionTo(request.getPermissionTo());
                    // Clear half day fields for permission
                    attendance.setHalfdayFrom(null);
                    attendance.setHalfdayTo(null);
                    attendance.setHalfdayType(null);

                    // Handle checkout time marking for permission
                    if (request.getMarkCheckoutTime() != null && request.getMarkCheckoutTime() &&
                            request.getCheckoutTime() != null) {
                        attendance.setCheckoutTime(request.getCheckoutTime());
                        checkoutMarked = true;
                        checkoutTimeStr = request.getCheckoutTime().toString();
                        System.out.println("Checkout time marked for permission: " + checkoutTimeStr);
                    }

                    System.out.println("Permission - From: " + request.getPermissionFrom() +
                            ", To: " + request.getPermissionTo());
                    break;

                case "Half Day Leave":
                    if (request.getHalfDayType() == null || request.getHalfDayType().trim().isEmpty()) {
                        response.put("status", "error");
                        response.put("message", "Half Day Type is required for Half Day Leave status");
                        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
                    }

                    if (request.getHalfDayFrom() == null || request.getHalfDayTo() == null) {
                        response.put("status", "error");
                        response.put("message", "Half day times are required for Half Day Leave status");
                        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
                    }

                    // Validate half day times
                    if (request.getHalfDayFrom().isAfter(request.getHalfDayTo()) ||
                            request.getHalfDayFrom().equals(request.getHalfDayTo())) {
                        response.put("status", "error");
                        response.put("message", "Half day 'To Time' must be after 'From Time'");
                        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
                    }

                    attendance.setHalfdayType(request.getHalfDayType());
                    attendance.setHalfdayFrom(request.getHalfDayFrom());
                    attendance.setHalfdayTo(request.getHalfDayTo());
                    // Clear permission fields for half day
                    attendance.setPermissionFrom(null);
                    attendance.setPermissionTo(null);

                    // Handle checkout time marking for half day
                    if (request.getMarkCheckoutTime() != null && request.getMarkCheckoutTime() &&
                            request.getCheckoutTime() != null) {
                        attendance.setCheckoutTime(request.getCheckoutTime());
                        checkoutMarked = true;
                        checkoutTimeStr = request.getCheckoutTime().toString();
                        System.out.println("Checkout time marked for half day: " + checkoutTimeStr);
                    }

                    System.out.println("Half Day Leave - Type: " + request.getHalfDayType() +
                            ", From: " + request.getHalfDayFrom() +
                            ", To: " + request.getHalfDayTo());
                    break;

                case "Absent":
                    // For "Absent", clear permission and half day fields but PRESERVE check-in data if exists
                    attendance.setPermissionFrom(null);
                    attendance.setPermissionTo(null);
                    attendance.setHalfdayFrom(null);
                    attendance.setHalfdayTo(null);
                    attendance.setHalfdayType(null);
                    // DON'T clear checkin/checkout for absent - let admin decide
                    break;

                case "On Duty":
                    // For "On Duty", clear permission and half day fields but PRESERVE checkin/checkout
                    attendance.setPermissionFrom(null);
                    attendance.setPermissionTo(null);
                    attendance.setHalfdayFrom(null);
                    attendance.setHalfdayTo(null);
                    attendance.setHalfdayType(null);
                    break;

                default:
                    // Clear permission and half day fields but PRESERVE existing checkin/checkout
                    attendance.setPermissionFrom(null);
                    attendance.setPermissionTo(null);
                    attendance.setHalfdayFrom(null);
                    attendance.setHalfdayTo(null);
                    attendance.setHalfdayType(null);
                    break;
            }

            // Save to database (update existing or save new)
            Attendance savedAttendance = attendanceRepository.save(attendance);

            // Debug log - Log what was saved
            System.out.println("=== SAVED TO DATABASE ===");
            System.out.println("Saved Attendance: " + savedAttendance.toString());
            System.out.println("Preserved Check-in Time: " + savedAttendance.getCheckinTime());
            if (checkoutMarked) {
                System.out.println("Checkout time saved: " + savedAttendance.getCheckoutTime());
            }

            // Success response
            response.put("status", "success");

            // Create detailed success message based on status type
            String successMessage = "Status updated successfully for " + employee.getUsername() +
                    " to '" + statusToStore + "' on " + request.getDate();

            if ("Permission".equals(statusToStore)) {
                successMessage += " (Permission: " + request.getPermissionFrom() + " - " + request.getPermissionTo() + ")";
            } else if ("Half Day Leave".equals(statusToStore)) {
                successMessage += " (Half Day: " + request.getHalfDayType() + " - " +
                        request.getHalfDayFrom() + " to " + request.getHalfDayTo() + ")";
            }

            // Add check-in preservation message
            if (savedAttendance.getCheckinTime() != null) {
                successMessage += ". Check-in time preserved: " + savedAttendance.getCheckinTime();
            }

            response.put("message", successMessage);

            // Add checkout information to response if applicable
            if (checkoutMarked) {
                response.put("checkoutMarked", "true");
                response.put("checkoutTime", checkoutTimeStr);
            }

            // Add override information to response
            if (statusOverridden) {
                response.put("statusOverridden", "true");
                response.put("previousStatus", previousStatus);
            }

            // Add preserved check-in info to response
            if (savedAttendance.getCheckinTime() != null) {
                response.put("checkinPreserved", "true");
                response.put("checkinTime", savedAttendance.getCheckinTime().toString());
            }

            // Debug log
            System.out.println("=== STATUS UPDATE SUCCESS ===");
            System.out.println("Employee: " + employee.getUsername());
            System.out.println("Date: " + request.getDate());
            System.out.println("Status stored in DB: " + statusToStore);
            System.out.println("Reason: " + request.getReason());
            System.out.println("Check-in preserved: " + (savedAttendance.getCheckinTime() != null));
            System.out.println("Checkout marked: " + checkoutMarked);
            System.out.println("Status overridden: " + statusOverridden);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            System.err.println("Error updating status: " + e.getMessage());
            e.printStackTrace();

            response.put("status", "error");
            response.put("message", "Error updating status: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
    // Optional: Add a helper method to validate status transitions
    private boolean isValidStatusTransition(String currentStatus, String newStatus) {
        // Define valid status transitions
        if ("On Leave".equals(currentStatus)) {
            // From "On Leave" (Absent), only allow changing to "On Duty" or staying "On Leave"
            return "On Duty".equals(newStatus) || "On Leave".equals(newStatus);
        }

        if ("On Duty".equals(currentStatus)) {
            // From "On Duty", allow any transition
            return true;
        }

        if ("Permission".equals(currentStatus)) {
            // From "Permission", allow any transition
            return true;
        }

        if ("Half Day Leave".equals(currentStatus)) {
            // From "Half Day Leave", allow any transition
            return true;
        }

        // Default: allow transition
        return true;
    }

    @GetMapping("/check-allocation/{employeeId}/{date}")
    public ResponseEntity<?> checkTodayAllocation(@PathVariable String employeeId, @PathVariable String date) {
        try {
            LocalDate checkDate = LocalDate.parse(date);

            // Check if employee has any allocation for the given date
            List<Attendance> existingAttendanceList = attendanceRepository
                    .findByEmployeeIdAndDate(employeeId, checkDate);

            if (!existingAttendanceList.isEmpty()) {
                Attendance attendance = existingAttendanceList.get(0); // Get the first record

                // Check if it has permission or half day allocation
                if ("Permission".equalsIgnoreCase(attendance.getStatus()) &&
                        attendance.getPermissionFrom() != null && attendance.getPermissionTo() != null) {

                    Map<String, Object> response = new HashMap<>();
                    response.put("hasAllocation", true);
                    response.put("allocationType", "Permission");
                    response.put("startTime", attendance.getPermissionFrom().toString());
                    response.put("endTime", attendance.getPermissionTo().toString());
                    response.put("reason", attendance.getReason());

                    return ResponseEntity.ok(response);

                } else if ("Half Day".equalsIgnoreCase(attendance.getStatus()) &&
                        attendance.getHalfdayFrom() != null && attendance.getHalfdayTo() != null) {

                    Map<String, Object> response = new HashMap<>();
                    response.put("hasAllocation", true);
                    response.put("allocationType", "Half Day");
                    response.put("startTime", attendance.getHalfdayFrom().toString());
                    response.put("endTime", attendance.getHalfdayTo().toString());
                    response.put("halfdayType", attendance.getHalfdayType());
                    response.put("reason", attendance.getReason());

                    return ResponseEntity.ok(response);
                }
            }

            // No allocation found
            Map<String, Object> response = new HashMap<>();
            response.put("hasAllocation", false);
            response.put("message", "No allocation found for this date");

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Error checking allocation: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    //21/6/25
    @GetMapping("/permission-status/{employeeId}/{yearMonth}")
    public ResponseEntity<?> getPermissionStatus(@PathVariable String employeeId, @PathVariable String yearMonth) {
        try {
            System.out.println("=== GETTING PERMISSION STATUS ===");
            System.out.println("Employee ID: " + employeeId);
            System.out.println("Year-Month: " + yearMonth);

            // Validate employee exists
            Optional<Employee> employeeOpt = employeeRepository.findById(employeeId);
            if (!employeeOpt.isPresent()) {
                return ResponseEntity.badRequest().body(Map.of("error", "Employee not found"));
            }

            PermissionStatusDTO status = attendanceService.getPermissionStatus(employeeId, yearMonth);
            return ResponseEntity.ok(status);

        } catch (Exception e) {
            System.err.println("Error getting permission status: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Error retrieving permission status: " + e.getMessage()));
        }
    }
//23/6/25

    //
    @PostMapping("/permission")
    public ResponseEntity<?> markPermission(@RequestBody FaceRecognitionEntry entry) {
        FaceRecognitionResult result = attendanceService.handlePermissionCheck(entry);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/halfday")
    public ResponseEntity<?> markHalfDay(@RequestBody FaceRecognitionEntry entry) {
        FaceRecognitionResult result = attendanceService.handleHalfDayCheck(entry);
        return ResponseEntity.ok(result);
    }

    // Add this method to your AttendanceController class

    @GetMapping("/status/{empId}/{date}")
    public ResponseEntity<Map<String, Object>> getAttendanceStatus(
            @PathVariable String empId,
            @PathVariable String date) {

        Map<String, Object> response = new HashMap<>();

        try {
            LocalDate checkDate = LocalDate.parse(date, DateTimeFormatter.ISO_LOCAL_DATE);

            // Check if employee exists
            Optional<Employee> employeeOpt = employeeRepository.findById(empId);
            if (!employeeOpt.isPresent()) {
                response.put("error", true);
                response.put("message", "Employee not found with ID: " + empId);
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(response);
            }

            Employee employee = employeeOpt.get();

            // Get attendance records for the date
            List<Attendance> attendanceRecords = attendanceRepository.findByEmployeeIdAndDate(empId, checkDate);

            if (attendanceRecords.isEmpty()) {
                response.put("hasAttendance", false);
                response.put("message", "No attendance record found for " + date);
                response.put("employeeId", empId);
                response.put("employeeName", employee.getUsername());
                response.put("date", date);
                response.put("status", "Not Marked");
                return ResponseEntity.ok(response);
            }

            Attendance attendance = attendanceRecords.get(0);
            String displayStatus = getDisplayStatus(attendance.getStatus());

            response.put("hasAttendance", true);
            response.put("employeeId", empId);
            response.put("employeeName", attendance.getEmployeeName());
            response.put("department", attendance.getDepartment());
            response.put("designation", attendance.getDesignation());
            response.put("date", date);
            response.put("status", displayStatus);
            response.put("reason", attendance.getReason());

            // Add check-in/check-out times if available
            if (attendance.getCheckinTime() != null) {
                response.put("checkinTime", attendance.getCheckinTime().toString());
            }
            if (attendance.getCheckoutTime() != null) {
                response.put("checkoutTime", attendance.getCheckoutTime().toString());
            }

            // Add permission details if applicable
            if ("Permission".equals(attendance.getStatus()) || "Permission Granted".equals(attendance.getStatus())) {
                if (attendance.getPermissionFrom() != null && attendance.getPermissionTo() != null) {
                    response.put("permissionFrom", attendance.getPermissionFrom().toString());
                    response.put("permissionTo", attendance.getPermissionTo().toString());
                }
            }

            // Add half day details if applicable
            if ("Half Day Leave".equals(attendance.getStatus())) {
                if (attendance.getHalfdayFrom() != null && attendance.getHalfdayTo() != null) {
                    response.put("halfdayType", attendance.getHalfdayType());
                    response.put("halfdayFrom", attendance.getHalfdayFrom().toString());
                    response.put("halfdayTo", attendance.getHalfdayTo().toString());
                }
            }

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            response.put("error", true);
            response.put("message", "Error fetching attendance status: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
    //29/6

    @PostMapping("/mark-holiday")
    public ResponseEntity<HolidayMarkingResult> markHoliday(@RequestBody HolidayRequest request) {
        try {
            HolidayMarkingResult result = attendanceService.markHolidayForAllEmployees(
                    request.getDate(),
                    request.getReason(),
                    request.getIsAdminOverride()
            );
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            System.err.println("Error marking holiday: " + e.getMessage());
            e.printStackTrace();
            HolidayMarkingResult errorResult = new HolidayMarkingResult();
            errorResult.setMessage("Error marking holiday: " + e.getMessage());
            errorResult.setEmployeesMarked(0);
            errorResult.setExistingRecordsUpdated(0);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResult);
        }
    }

}