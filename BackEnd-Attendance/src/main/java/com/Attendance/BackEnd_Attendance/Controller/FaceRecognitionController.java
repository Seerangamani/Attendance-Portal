package com.Attendance.BackEnd_Attendance.Controller;

import com.Attendance.BackEnd_Attendance.Model.*;
import com.Attendance.BackEnd_Attendance.Repository.AttendanceRepository;
import com.Attendance.BackEnd_Attendance.Repository.EmployeeRepository;
import com.Attendance.BackEnd_Attendance.Service.AttendanceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = {"http://127.0.0.1:5500", "http://localhost:5500"}, allowCredentials = "true")
public class FaceRecognitionController {

    @Autowired
    private EmployeeRepository employeeRepository;

    @Autowired
    private AttendanceService attendanceService;

    @Autowired
    private AttendanceRepository attendanceRepository;

    private final RestTemplate restTemplate = new RestTemplate();

    // Updated face recognition method to include gender-based timing info in response
    @PostMapping("/face-recognition")
    public ResponseEntity<?> recognizeFace(@RequestBody FaceRecognitionRequest request) {
        String empId = request.getEmployeeId();
        String capturedImage = request.getLiveImage();

        // Validate input
        if (capturedImage == null || capturedImage.trim().isEmpty()) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Captured image is required.");
            errorResponse.put("status", "invalid_input");
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
        }

        if (empId.equalsIgnoreCase("Pub01") || empId.equalsIgnoreCase("Public")) {
            List<Employee> allEmployees = employeeRepository.findAll();

            // Log for debugging
            System.out.println("Public recognition: Found " + allEmployees.size() + " employees");

            for (Employee employee : allEmployees) {
                byte[] storedImageBytes = employee.getProfileImage();

                if (storedImageBytes != null && storedImageBytes.length > 0) {
                    System.out.println("Checking face match for employee: " + employee.getId() + " - " + employee.getUsername());

                    boolean matchResult = isFaceMatched(capturedImage, storedImageBytes);
                    System.out.println("Face match result for " + employee.getId() + ": " + matchResult);

                    if (matchResult) {
                        // Check if employee has permission or half-day allocation for today
                        boolean shouldMarkAttendance = shouldMarkAttendanceForEmployee(employee.getId());

                        ResponseEntity<?> attendanceResponse = null;
                        boolean attendanceMarked = false;
                        String attendanceMessage = "";

                        if (shouldMarkAttendance) {
                            // Mark attendance only if no permission/half-day allocation exists
                            attendanceResponse = markAttendance(employee, "Checkin");
                            attendanceMarked = attendanceResponse.getStatusCode().is2xxSuccessful();
                            attendanceMessage = attendanceMarked ? "Check-in time added successfully" : "Attendance marking failed";
                        } else {
                            attendanceMessage = "Check-in not added - Employee has Permission or Half Day Leave allocation";
                        }

                        // Create response with employee data
                        Map<String, Object> response = new HashMap<>();
                        response.put("status", "matched");
                        response.put("message", "Face matched successfully");

                        Map<String, String> employeeData = new HashMap<>();
                        employeeData.put("id", employee.getId());
                        employeeData.put("name", employee.getUsername());
                        employeeData.put("department", employee.getDepartment());
                        employeeData.put("gender", employee.getGender());
                        employeeData.put("workTiming", getGenderBasedTimingInfo(employee.getGender()));
                        employeeData.put("faceImage", Base64.getEncoder().encodeToString(storedImageBytes));

                        response.put("employee", employeeData);
                        response.put("attendanceMarked", attendanceMarked);
                        response.put("attendanceMessage", attendanceMessage);

                        return ResponseEntity.ok(response);
                    }
                } else {
                    System.out.println("No profile image found for employee: " + employee.getId());
                }
            }

            System.out.println("No face match found for any employee");
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Face not matched with any employee. Please ensure proper lighting and face positioning.");
            errorResponse.put("status", "not_matched");
            errorResponse.put("debug_info", "Checked " + allEmployees.size() + " employees");
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
        }

        // For specific employee ID
        Optional<Employee> empOpt = employeeRepository.findById(empId);
        if (empOpt.isEmpty()) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Employee not found with ID: " + empId);
            errorResponse.put("status", "not_found");
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(errorResponse);
        }

        Employee employee = empOpt.get();
        byte[] storedImageBytes = employee.getProfileImage();

        if (storedImageBytes == null || storedImageBytes.length == 0) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "No profile image found for employee: " + employee.getUsername());
            errorResponse.put("status", "no_image");
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
        }

        System.out.println("Attempting face match for specific employee: " + empId + " - " + employee.getUsername());
        boolean matched = isFaceMatched(capturedImage, storedImageBytes);
        System.out.println("Face match result for " + empId + ": " + matched);

        if (matched) {
            // Check if employee has permission or half-day allocation for today
            boolean shouldMarkAttendance = shouldMarkAttendanceForEmployee(employee.getId());

            boolean attendanceMarked = false;
            String attendanceMessage = "";

            if (shouldMarkAttendance) {
                // Mark attendance only if no permission/half-day allocation exists
                ResponseEntity<?> attendanceResponse = markAttendance(employee, "Checkin");
                attendanceMarked = attendanceResponse.getStatusCode().is2xxSuccessful();
                attendanceMessage = attendanceMarked ? "Check-in time added successfully" : "Attendance marking failed";
            } else {
                attendanceMessage = "Check-in not added - Employee has Permission or Half Day Leave allocation";
            }

            // Return success response with employee data
            Map<String, Object> response = new HashMap<>();
            response.put("status", "matched");
            response.put("message", "Face matched successfully");

            Map<String, String> employeeData = new HashMap<>();
            employeeData.put("id", employee.getId());
            employeeData.put("name", employee.getUsername());
            employeeData.put("department", employee.getDepartment());
            employeeData.put("gender", employee.getGender());
            employeeData.put("workTiming", getGenderBasedTimingInfo(employee.getGender()));
            employeeData.put("faceImage", Base64.getEncoder().encodeToString(storedImageBytes));

            response.put("employee", employeeData);
            response.put("attendanceMarked", attendanceMarked);
            response.put("attendanceMessage", attendanceMessage);

            return ResponseEntity.ok(response);
        } else {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Face does not match with " + employee.getUsername() + ". Please ensure proper lighting and face positioning.");
            errorResponse.put("status", "not_matched");
            errorResponse.put("employeeName", employee.getUsername());
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
        }
    }

    // Helper method to check if attendance should be marked
    private boolean shouldMarkAttendanceForEmployee(String employeeId) {
        try {
            LocalDate today = LocalDate.now();
            List<Attendance> todayAttendance = attendanceRepository.findByEmployeeIdAndDate(employeeId, today);

            if (todayAttendance.isEmpty()) {
                // No attendance record exists, safe to mark attendance
                return true;
            }

            Attendance attendance = todayAttendance.get(0);

            // Check if employee has Permission allocation
            if ("Permission".equals(attendance.getStatus()) || "Permission Granted".equals(attendance.getStatus())) {
                if (attendance.getPermissionFrom() != null && attendance.getPermissionTo() != null) {
                    System.out.println("Employee " + employeeId + " has Permission allocation - Check-in will not be added");
                    return false;
                }
            }

            // Check if employee has Half Day Leave allocation
            if ("Half Day Leave".equals(attendance.getStatus())) {
                if (attendance.getHalfdayFrom() != null && attendance.getHalfdayTo() != null) {
                    System.out.println("Employee " + employeeId + " has Half Day Leave allocation - Check-in will not be added");
                    return false;
                }
            }

            // If attendance record exists but no permission/half-day times, allow check-in update
            return true;

        } catch (Exception e) {
            System.err.println("Error checking attendance allocation for employee " + employeeId + ": " + e.getMessage());
            // In case of error, default to allowing attendance marking
            return true;
        }
    }
    @PostMapping("/face-recognition/checkout")
    public ResponseEntity<?> recognizeFaceForCheckout(@RequestBody FaceRecognitionRequest request) {
        String empId = request.getEmployeeId();
        String capturedImage = request.getLiveImage();

        if (empId.equalsIgnoreCase("Pub01") || empId.equalsIgnoreCase("Public")) {
            List<Employee> allEmployees = employeeRepository.findAll();

            for (Employee employee : allEmployees) {
                byte[] storedImageBytes = employee.getProfileImage();

                if (storedImageBytes != null && isFaceMatched(capturedImage, storedImageBytes)) {
                    // ✅ NEW: Check gender-based timing validation before proceeding
                    Map<String, Object> timingValidation = validateCheckoutTiming(employee.getId(), employee.getGender());
                    if (!(Boolean) timingValidation.get("allowed")) {
                        Map<String, Object> errorResponse = new HashMap<>();
                        errorResponse.put("status", "timing_error");
                        errorResponse.put("message", timingValidation.get("message"));

                        // Include employee data for frontend display
                        Map<String, String> employeeData = new HashMap<>();
                        employeeData.put("id", employee.getId());
                        employeeData.put("name", employee.getUsername());
                        employeeData.put("department", employee.getDepartment());
                        employeeData.put("gender", employee.getGender());
                        employeeData.put("faceImage", Base64.getEncoder().encodeToString(storedImageBytes));

                        errorResponse.put("employee", employeeData);
                        errorResponse.put("earliestCheckoutTime", timingValidation.get("earliestTime"));
                        errorResponse.put("currentTime", timingValidation.get("currentTime"));

                        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
                    }

                    // Face matched and timing is valid - proceed with checkout
                    try {
                        AttendanceRequest checkoutRequest = new AttendanceRequest();
                        checkoutRequest.setEmployeeId(employee.getId());
                        checkoutRequest.setEmployeeName(employee.getUsername());
                        checkoutRequest.setDepartment(employee.getDepartment());
                        checkoutRequest.setTime(OffsetDateTime.now().toString());

                        String checkoutResult = attendanceService.recordCheckout(checkoutRequest);

                        Map<String, Object> response = new HashMap<>();
                        response.put("status", "matched");
                        response.put("message", "Face matched and checkout recorded successfully");
                        response.put("checkoutResult", checkoutResult);

                        Map<String, String> employeeData = new HashMap<>();
                        employeeData.put("id", employee.getId());
                        employeeData.put("name", employee.getUsername());
                        employeeData.put("department", employee.getDepartment());
                        employeeData.put("gender", employee.getGender());
                        employeeData.put("workTiming", getGenderBasedTimingInfo(employee.getGender()));
                        employeeData.put("faceImage", Base64.getEncoder().encodeToString(storedImageBytes));

                        response.put("employee", employeeData);
                        return ResponseEntity.ok(response);

                    } catch (Exception e) {
                        // Handle checkout errors (early leave, already checked out, etc.)
                        Map<String, Object> errorResponse = new HashMap<>();
                        errorResponse.put("status", "checkout_error");
                        errorResponse.put("message", e.getMessage());

                        // Still include employee data for frontend use
                        Map<String, String> employeeData = new HashMap<>();
                        employeeData.put("id", employee.getId());
                        employeeData.put("name", employee.getUsername());
                        employeeData.put("department", employee.getDepartment());
                        errorResponse.put("employee", employeeData);

                        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
                    }
                }
            }

            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Face not matched with any employee.");
            errorResponse.put("status", "not_matched");
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
        }

        // For specific employee ID
        Optional<Employee> empOpt = employeeRepository.findById(empId);
        if (empOpt.isEmpty()) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Employee not found.");
            errorResponse.put("status", "not_found");
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(errorResponse);
        }

        Employee employee = empOpt.get();
        byte[] storedImageBytes = employee.getProfileImage();

        if (storedImageBytes == null || storedImageBytes.length == 0) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Stored face image not found.");
            errorResponse.put("status", "no_image");
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
        }

        boolean matched = isFaceMatched(capturedImage, storedImageBytes);
        if (matched) {
            // ✅ NEW: Check gender-based timing validation before proceeding
            Map<String, Object> timingValidation = validateCheckoutTiming(employee.getId(), employee.getGender());
            if (!(Boolean) timingValidation.get("allowed")) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("status", "timing_error");
                errorResponse.put("message", timingValidation.get("message"));

                // Include employee data for frontend display
                Map<String, String> employeeData = new HashMap<>();
                employeeData.put("id", employee.getId());
                employeeData.put("name", employee.getUsername());
                employeeData.put("department", employee.getDepartment());
                employeeData.put("gender", employee.getGender());
                employeeData.put("faceImage", Base64.getEncoder().encodeToString(storedImageBytes));

                errorResponse.put("employee", employeeData);
                errorResponse.put("earliestCheckoutTime", timingValidation.get("earliestTime"));
                errorResponse.put("currentTime", timingValidation.get("currentTime"));

                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
            }

            // Face matched and timing is valid - proceed with checkout
            try {
                AttendanceRequest checkoutRequest = new AttendanceRequest();
                checkoutRequest.setEmployeeId(employee.getId());
                checkoutRequest.setEmployeeName(employee.getUsername());
                checkoutRequest.setDepartment(employee.getDepartment());
                checkoutRequest.setTime(OffsetDateTime.now().toString());

                String checkoutResult = attendanceService.recordCheckout(checkoutRequest);

                Map<String, Object> response = new HashMap<>();
                response.put("status", "matched");
                response.put("message", "Face matched and checkout recorded successfully");
                response.put("checkoutResult", checkoutResult);

                Map<String, String> employeeData = new HashMap<>();
                employeeData.put("id", employee.getId());
                employeeData.put("name", employee.getUsername());
                employeeData.put("department", employee.getDepartment());
                employeeData.put("gender", employee.getGender());
                employeeData.put("workTiming", getGenderBasedTimingInfo(employee.getGender()));
                employeeData.put("faceImage", Base64.getEncoder().encodeToString(storedImageBytes));

                response.put("employee", employeeData);
                return ResponseEntity.ok(response);

            } catch (Exception e) {
                // Handle checkout errors (early leave, already checked out, etc.)
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("status", "checkout_error");
                errorResponse.put("message", e.getMessage());

                // Still include employee data for frontend use
                Map<String, String> employeeData = new HashMap<>();
                employeeData.put("id", employee.getId());
                employeeData.put("name", employee.getUsername());
                employeeData.put("department", employee.getDepartment());
                errorResponse.put("employee", employeeData);

                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
            }
        } else {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Face mismatch. Unauthorized.");
            errorResponse.put("status", "not_matched");
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
        }
    }

    private boolean isFaceMatched(String capturedImage, byte[] storedImageBytes) {
        try {
            // Validate inputs
            if (capturedImage == null || capturedImage.trim().isEmpty()) {
                System.err.println("Captured image is null or empty");
                return false;
            }

            if (storedImageBytes == null || storedImageBytes.length == 0) {
                System.err.println("Stored image bytes are null or empty");
                return false;
            }

            String storedBase64 = Base64.getEncoder().encodeToString(storedImageBytes);

            // Validate base64 strings
            if (storedBase64.isEmpty()) {
                System.err.println("Failed to encode stored image to Base64");
                return false;
            }

            Map<String, String> body = new HashMap<>();
            body.put("capturedImage", capturedImage);
            body.put("storedImage", storedBase64);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Connection", "close"); // Prevent connection issues

            HttpEntity<Map<String, String>> entity = new HttpEntity<>(body, headers);

            System.out.println("Sending face comparison request to Python service...");

            // Add timeout configuration
            restTemplate.getMessageConverters().add(0, new org.springframework.http.converter.StringHttpMessageConverter(java.nio.charset.StandardCharsets.UTF_8));

            ResponseEntity<Map> response = restTemplate.postForEntity(
                    "http://localhost:5000/compare",
                    entity,
                    Map.class
            );

            System.out.println("Face comparison response status: " + response.getStatusCode());

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Object matchValue = response.getBody().get("match");
                Object confidenceValue = response.getBody().get("confidence");

                System.out.println("Match result: " + matchValue);
                if (confidenceValue != null) {
                    System.out.println("Confidence: " + confidenceValue);
                }

                return Boolean.TRUE.equals(matchValue);
            } else {
                System.err.println("Face comparison service returned non-OK status: " + response.getStatusCode());
            }

        } catch (org.springframework.web.client.ResourceAccessException e) {
            System.err.println("Cannot connect to face comparison service at localhost:5000: " + e.getMessage());
            System.err.println("Please ensure Python face recognition service is running on port 5000");
        } catch (org.springframework.web.client.RestClientException e) {
            System.err.println("REST client error during face comparison: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error comparing faces: " + e.getMessage());
            e.printStackTrace();
        }

        return false;
    }
    // Updated markAttendance method in FaceRecognitionController
    private ResponseEntity<?> markAttendance(Employee employee, String status) {
        try {
            String empName = Optional.ofNullable(employee.getUsername()).orElse("Unknown");
            String department = Optional.ofNullable(employee.getDepartment()).orElse("Unknown");

            // Create attendance map for gender-based timing check
            Map<String, String> attendanceMap = new HashMap<>();
            attendanceMap.put("employeeId", employee.getId());
            attendanceMap.put("employeeName", empName);
            attendanceMap.put("department", department);
            attendanceMap.put("time", OffsetDateTime.now().toString());
            attendanceMap.put("status", status);

            // Use AttendanceService.checkIn() which already handles gender-based timing
            Attendance attendance = attendanceService.checkIn(attendanceMap);

            // Success response with gender-based timing info
            Map<String, Object> response = new HashMap<>();
            response.put("message", "Face matched. Attendance marked with gender-based timing.");
            response.put("employeeName", empName);
            response.put("employeeGender", employee.getGender());

            // Add timing info based on gender
            String timingInfo = getGenderBasedTimingInfo(employee.getGender());
            response.put("workTiming", timingInfo);
            response.put("attendanceStatus", attendance.getStatus());
            response.put("checkinTime", attendance.getCheckinTime().toString());
            response.put("status", "matched");

            return ResponseEntity.ok(response);

        } catch (IllegalArgumentException e) {
            // Handle cases like "already checked in"
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", e.getMessage());
            errorResponse.put("status", "error");
            errorResponse.put("employeeName", employee.getUsername());
            return ResponseEntity.status(HttpStatus.CONFLICT).body(errorResponse);
        } catch (Exception e) {
            System.err.println("Error marking attendance: " + e.getMessage());
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Error marking attendance: " + e.getMessage());
            errorResponse.put("status", "error");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @GetMapping("/employees/face-images")
    public ResponseEntity<?> getAllEmployeeFaceImages() {
        List<Employee> employees = employeeRepository.findAll();

        List<Map<String, String>> faceList = new ArrayList<>();

        for (Employee emp : employees) {
            byte[] imageBytes = emp.getProfileImage();

            if (imageBytes != null && imageBytes.length > 0) {
                String base64Image = Base64.getEncoder().encodeToString(imageBytes);
                Map<String, String> faceMap = new HashMap<>();
                faceMap.put("id", emp.getId());
                faceMap.put("name", emp.getUsername());
                faceMap.put("department", emp.getDepartment());
                faceMap.put("base64Image", base64Image);

                faceList.add(faceMap);
            }
        }

        return ResponseEntity.ok(faceList);
    }

    @GetMapping("/employee/face-image/{id}")
    public ResponseEntity<?> getEmployeeFaceImage(@PathVariable String id) {
        Optional<Employee> empOpt = employeeRepository.findById(id);
        if (empOpt.isPresent()) {
            byte[] imageBytes = empOpt.get().getProfileImage();
            if (imageBytes != null) {
                Map<String, String> result = new HashMap<>();
                result.put("base64Image", Base64.getEncoder().encodeToString(imageBytes));
                return ResponseEntity.ok(result);
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of("message", "Image not found"));
            }
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of("message", "Employee not found"));
        }
    }
    // Helper method to provide timing information based on gender
    private String getGenderBasedTimingInfo(String gender) {
        if ("FEMALE".equalsIgnoreCase(gender) || "F".equalsIgnoreCase(gender)) {
            return "Female Employee: 9:30 AM - 5:00 PM";
        } else {
            return "Male Employee: 9:30 AM - 6:00 PM";
        }
    }

    // New method to handle face recognition checkout with gender-based validation
    @PostMapping("/face-recognition/process-checkout")
    public ResponseEntity<?> processFaceRecognitionCheckout(@RequestBody Map<String, String> request) {
        try {
            String employeeId = request.get("employeeId");

            if (employeeId == null) {
                Map<String, String> errorResponse = new HashMap<>();
                errorResponse.put("message", "Employee ID is required");
                errorResponse.put("status", "error");
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
            }

            // Validate employee exists and get gender info
            Optional<Employee> empOpt = employeeRepository.findById(employeeId);
            if (empOpt.isEmpty()) {
                Map<String, String> errorResponse = new HashMap<>();
                errorResponse.put("message", "Employee not found.");
                errorResponse.put("status", "not_found");
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(errorResponse);
            }

            Employee employee = empOpt.get();

            // ✅ NEW: Check gender-based timing validation before proceeding
            Map<String, Object> timingValidation = validateCheckoutTiming(employee.getId(), employee.getGender());

            if (!(Boolean) timingValidation.get("allowed")) {
                Map<String, Object> errorResponse = new HashMap<>();
                errorResponse.put("status", "timing_error");
                errorResponse.put("message", timingValidation.get("message"));
                errorResponse.put("employeeName", employee.getUsername());
                errorResponse.put("employeeGender", employee.getGender());
                errorResponse.put("earliestCheckoutTime", timingValidation.get("earliestTime"));
                errorResponse.put("currentTime", timingValidation.get("currentTime"));
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
            }

            // Create checkout request with current time
            AttendanceRequest checkoutRequest = new AttendanceRequest();
            checkoutRequest.setEmployeeId(employeeId);
            checkoutRequest.setEmployeeName(employee.getUsername());
            checkoutRequest.setDepartment(employee.getDepartment());
            checkoutRequest.setTime(OffsetDateTime.now().toString());

            // Use AttendanceService.recordCheckout() which handles gender-based validation
            String result = attendanceService.recordCheckout(checkoutRequest);

            Map<String, Object> response = new HashMap<>();

            if (result.contains("Successfully checked out")) {
                response.put("status", "success");
                response.put("message", result);
                response.put("employeeName", employee.getUsername());
                response.put("employeeGender", employee.getGender());
                response.put("workTiming", getGenderBasedTimingInfo(employee.getGender()));

                // Extract checkout time from result if available
                if (result.contains("at ")) {
                    String[] parts = result.split("at ");
                    if (parts.length > 1) {
                        String timeInfo = parts[1].split("\\.")[0];
                        response.put("checkoutTime", timeInfo);
                    }
                }

                return ResponseEntity.ok(response);
            } else {
                // Handle checkout errors (early leave, no checkin, etc.)
                response.put("status", "error");
                response.put("message", result);
                response.put("employeeName", employee.getUsername());
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(response);
            }

        } catch (Exception e) {
            System.err.println("Error processing face recognition checkout: " + e.getMessage());
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("message", "Error processing checkout: " + e.getMessage());
            errorResponse.put("status", "error");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @PostMapping("/face-recognition-with-allocation")
    public ResponseEntity<?> recognizeFaceWithAllocationCheck(@RequestBody FaceRecognitionRequest request) {
        String empId = request.getEmployeeId();
        String capturedImage = request.getLiveImage();

        // First perform face recognition
        ResponseEntity<?> faceResult = recognizeFace(request);

        if (!faceResult.getStatusCode().is2xxSuccessful()) {
            return faceResult; // Return face recognition error
        }

        try {
            // Check for today's allocation
            LocalDate today = LocalDate.now();
            List<Attendance> allocationList = attendanceRepository
                    .findByEmployeeIdAndDate(empId, today);

            Map<String, Object> response = new HashMap<>();
            response.putAll((Map<String, Object>) faceResult.getBody());

            if (!allocationList.isEmpty()) {
                // Get the first (or most recent) allocation for today
                Attendance allocation = allocationList.get(0);

                if ("Permission".equals(allocation.getStatus()) &&
                        allocation.getPermissionFrom() != null) {

                    response.put("hasAllocation", true);
                    response.put("allocationType", "Permission");
                    response.put("startTime", allocation.getPermissionFrom().toString());
                    response.put("endTime", allocation.getPermissionTo().toString());
                    response.put("reason", allocation.getReason());

                } else if ("Half Day".equals(allocation.getStatus()) &&
                        allocation.getHalfdayFrom() != null) {

                    response.put("hasAllocation", true);
                    response.put("allocationType", "Half Day");
                    response.put("startTime", allocation.getHalfdayFrom().toString());
                    response.put("endTime", allocation.getHalfdayTo().toString());
                    response.put("halfdayType", allocation.getHalfdayType());
                    response.put("reason", allocation.getReason());
                } else {
                    response.put("hasAllocation", false);
                }
            } else {
                response.put("hasAllocation", false);
            }

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            // If allocation check fails, return face recognition success with error note
            Map<String, Object> response = new HashMap<>();
            response.putAll((Map<String, Object>) faceResult.getBody());
            response.put("hasAllocation", false);
            response.put("allocationError", "Could not check allocation: " + e.getMessage());

            return ResponseEntity.ok(response);
        }
    }

    //28/6/25
    // ✅ NEW HELPER METHOD: Validate checkout timing based on gender
    private Map<String, Object> validateCheckoutTiming(String employeeId, String gender) {
        Map<String, Object> result = new HashMap<>();
        LocalTime now = LocalTime.now();
        LocalTime earliestCheckoutTime;
        String timingInfo;

        // Special case for employee mmw027 with custom timing (9:40 AM - 4:00 PM)
        if ("mmw027".equalsIgnoreCase(employeeId)) {
            earliestCheckoutTime = LocalTime.of(16, 0); // 4:00 PM
            timingInfo = "Special Employee Timing: 9:40 AM - 4:00 PM";

            // Check for permission or half-day allocations for mmw027
            LocalDate today = LocalDate.now();

            // Check for permission allocation
            Optional<Attendance> permissionAllocation = attendanceRepository.findPermissionAllocation(employeeId, today);
            if (permissionAllocation.isPresent()) {
                LocalTime permissionTo = permissionAllocation.get().getPermissionTo();

                // Allow checkout if permission ends at 2:00-4:00 PM (special employee end time range)
                if (permissionTo != null &&
                        (permissionTo.equals(LocalTime.of(14, 0)) || permissionTo.equals(LocalTime.of(15, 0)) ||
                                permissionTo.equals(LocalTime.of(16, 0)))) {

                    result.put("allowed", true);
                    result.put("message", "Checkout allowed - Special employee with permission ending at work time.");
                    result.put("timingInfo", "Permission Time: " + permissionAllocation.get().getPermissionFrom() + " - " + permissionTo);
                    result.put("employeeType", "special_with_permission");
                    return result;
                }
            }

            // Check for half-day allocation
            Optional<Attendance> halfDayAllocation = attendanceRepository.findHalfDayAllocation(employeeId, today);
            if (halfDayAllocation.isPresent()) {
                LocalTime halfDayTo = halfDayAllocation.get().getHalfdayTo();

                // Allow checkout if half-day is second half (1:30 PM - 4:00 PM)
                if (halfDayTo != null && halfDayTo.equals(LocalTime.of(16, 0))) {
                    LocalTime halfDayFrom = halfDayAllocation.get().getHalfdayFrom();
                    if (halfDayFrom != null && halfDayFrom.equals(LocalTime.of(13, 30))) {
                        result.put("allowed", true);
                        result.put("message", "Checkout allowed - Special employee with second half as half-day.");
                        result.put("timingInfo", "Half Day Time: " + halfDayFrom + " - " + halfDayTo);
                        result.put("employeeType", "special_with_halfday");
                        return result;
                    }
                }
            }

            // Regular timing validation for mmw027
            if (now.isBefore(earliestCheckoutTime)) {
                result.put("allowed", false);
                result.put("message", "Checkout not allowed. Special employee timing requires you to work until 4:00 PM.");
                result.put("earliestTime", earliestCheckoutTime.toString());
                result.put("currentTime", now.toString());
                result.put("timingInfo", timingInfo);
                result.put("employeeType", "special");
                return result;
            }

            result.put("allowed", true);
            result.put("message", "Checkout allowed for special employee timing.");
            result.put("timingInfo", timingInfo);
            result.put("employeeType", "special");
            return result;
        }

        // Standard gender-based timing for other employees
        LocalDate today = LocalDate.now();

        if (gender != null && (gender.equalsIgnoreCase("FEMALE") || gender.equalsIgnoreCase("F"))) {
            // Female employees: 9:30 AM - 5:00 PM (7.5 hours)
            earliestCheckoutTime = LocalTime.of(17, 0); // 5:00 PM
            timingInfo = "Female Employee Timing: 9:30 AM - 5:00 PM";

            // Check for permission allocation for female employees
            Optional<Attendance> permissionAllocation = attendanceRepository.findPermissionAllocation(employeeId, today);
            if (permissionAllocation.isPresent()) {
                LocalTime permissionTo = permissionAllocation.get().getPermissionTo();

                // Allow checkout if permission ends at 3:00-5:00 PM or 4:00-5:00 PM (female work end time range)
                if (permissionTo != null &&
                        (permissionTo.equals(LocalTime.of(15, 0)) || permissionTo.equals(LocalTime.of(16, 0)) ||
                                permissionTo.equals(LocalTime.of(17, 0)))) {

                    result.put("allowed", true);
                    result.put("message", "Checkout allowed - Female employee with permission ending at work time.");
                    result.put("timingInfo", "Permission Time: " + permissionAllocation.get().getPermissionFrom() + " - " + permissionTo);
                    result.put("employeeType", "female_with_permission");
                    return result;
                }
            }

            // Check for half-day allocation for female employees
            Optional<Attendance> halfDayAllocation = attendanceRepository.findHalfDayAllocation(employeeId, today);
            if (halfDayAllocation.isPresent()) {
                LocalTime halfDayTo = halfDayAllocation.get().getHalfdayTo();

                // Allow checkout if half-day is second half (1:30 PM - 5:00 PM)
                if (halfDayTo != null && halfDayTo.equals(LocalTime.of(17, 0))) {
                    LocalTime halfDayFrom = halfDayAllocation.get().getHalfdayFrom();
                    if (halfDayFrom != null && halfDayFrom.equals(LocalTime.of(13, 30))) {
                        result.put("allowed", true);
                        result.put("message", "Checkout allowed - Female employee with second half as half-day.");
                        result.put("timingInfo", "Half Day Time: " + halfDayFrom + " - " + halfDayTo);
                        result.put("employeeType", "female_with_halfday");
                        return result;
                    }
                }
            }

        } else {
            // Male employees: 9:30 AM - 6:00 PM (8.5 hours)
            earliestCheckoutTime = LocalTime.of(18, 0); // 6:00 PM
            timingInfo = "Male Employee Timing: 9:30 AM - 6:00 PM";

            // Check for permission allocation for male employees
            Optional<Attendance> permissionAllocation = attendanceRepository.findPermissionAllocation(employeeId, today);
            if (permissionAllocation.isPresent()) {
                LocalTime permissionTo = permissionAllocation.get().getPermissionTo();

                // Allow checkout if permission ends at 4:00-6:00 PM or 5:00-6:00 PM (male work end time range)
                if (permissionTo != null &&
                        (permissionTo.equals(LocalTime.of(16, 0)) || permissionTo.equals(LocalTime.of(17, 0)) ||
                                permissionTo.equals(LocalTime.of(18, 0)))) {

                    result.put("allowed", true);
                    result.put("message", "Checkout allowed - Male employee with permission ending at work time.");
                    result.put("timingInfo", "Permission Time: " + permissionAllocation.get().getPermissionFrom() + " - " + permissionTo);
                    result.put("employeeType", "male_with_permission");
                    return result;
                }
            }

            // Check for half-day allocation for male employees
            Optional<Attendance> halfDayAllocation = attendanceRepository.findHalfDayAllocation(employeeId, today);
            if (halfDayAllocation.isPresent()) {
                LocalTime halfDayTo = halfDayAllocation.get().getHalfdayTo();

                // Allow checkout if half-day is second half (1:30 PM - 6:00 PM)
                if (halfDayTo != null && halfDayTo.equals(LocalTime.of(18, 0))) {
                    LocalTime halfDayFrom = halfDayAllocation.get().getHalfdayFrom();
                    if (halfDayFrom != null && halfDayFrom.equals(LocalTime.of(13, 30))) {
                        result.put("allowed", true);
                        result.put("message", "Checkout allowed - Male employee with second half as half-day.");
                        result.put("timingInfo", "Half Day Time: " + halfDayFrom + " - " + halfDayTo);
                        result.put("employeeType", "male_with_halfday");
                        return result;
                    }
                }
            }
        }

        // Regular timing validation if no special permissions/half-days
        if (now.isBefore(earliestCheckoutTime)) {
            long minutesUntilCheckout = ChronoUnit.MINUTES.between(now, earliestCheckoutTime);
            long hoursUntilCheckout = minutesUntilCheckout / 60;
            long remainingMinutes = minutesUntilCheckout % 60;

            String waitMessage;
            if (hoursUntilCheckout > 0) {
                waitMessage = String.format("Please wait %d hours and %d minutes until your scheduled checkout time.",
                        hoursUntilCheckout, remainingMinutes);
            } else {
                waitMessage = String.format("Please wait %d minutes until your scheduled checkout time.",
                        remainingMinutes);
            }

            result.put("allowed", false);
            result.put("message", "Checkout not allowed due to gender-based timing restrictions. " + waitMessage);
            result.put("earliestTime", earliestCheckoutTime.toString());
            result.put("currentTime", now.toString());
            result.put("timingInfo", timingInfo);
            result.put("employeeType", "standard");
            return result;
        }

        // Allow checkout for regular timing (after work hours)
        result.put("allowed", true);
        result.put("message", "Checkout allowed.");
        result.put("timingInfo", timingInfo);
        result.put("employeeType", "standard");
        return result;
    }
}