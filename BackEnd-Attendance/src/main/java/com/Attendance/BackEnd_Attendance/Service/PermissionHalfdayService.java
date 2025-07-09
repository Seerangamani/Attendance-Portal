//package com.Attendance.BackEnd_Attendance.Service;
//
//import com.Attendance.BackEnd_Attendance.Model.Attendance;
//import com.Attendance.BackEnd_Attendance.Model.Employee;
//import com.Attendance.BackEnd_Attendance.Repository.PermissionHalfdayRepository;
//import com.Attendance.BackEnd_Attendance.Repository.EmployeeRepository;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.stereotype.Service;
//import org.springframework.web.multipart.MultipartFile;
//
//import java.time.LocalDate;
//import java.time.LocalTime;
//import java.time.format.DateTimeFormatter;
//import java.util.*;
//
//@Service
//public class PermissionHalfdayService {
//
//    @Autowired
//    private PermissionHalfdayRepository permissionHalfdayRepository;
//
//    @Autowired
//    private EmployeeRepository employeeRepository;
//
//    public Map<String, Object> processFaceRecognition(MultipartFile image, String checkType,
//                                                      String timestamp, String currentTime) {
//
//        Map<String, Object> response = new HashMap<>();
//
//        try {
//            // Step 1: Perform face recognition
//            Employee recognizedEmployee = performFaceRecognition(image);
//
//            if (recognizedEmployee == null) {
//                response.put("status", "face_not_recognized");
//                response.put("message", "Face not recognized in the system");
//                return response;
//            }
//
//            // Step 2: Check permission/halfday for today
//            LocalDate today = LocalDate.now();
//            Attendance todayAttendance = permissionHalfdayRepository
//                    .findByEmployeeIdAndDate(recognizedEmployee.getId(), today);
//
//            if (todayAttendance == null) {
//                response.put("status", "no_permission");
//                response.put("message", "No " + checkType + " permission found for today");
//                response.put("employee", createEmployeeResponse(recognizedEmployee));
//                return response;
//            }
//
//            // Step 3: Validate time and update status
//            LocalTime currentLocalTime = LocalTime.parse(currentTime, DateTimeFormatter.ofPattern("HH:mm:ss"));
//            String validationResult = validateTimeAndUpdateStatus(todayAttendance, checkType, currentLocalTime);
//
//            // Step 4: Save updated attendance
//            permissionHalfdayRepository.save(todayAttendance);
//
//            // Step 5: Prepare response
//            response.put("status", validationResult);
//            response.put("employee", createEmployeeResponse(recognizedEmployee));
//            response.put("attendance", createAttendanceResponse(todayAttendance));
//            response.put("timeInfo", createTimeInfo(currentLocalTime));
//
//            // Add appropriate message
//            switch (validationResult) {
//                case "success":
//                    response.put("message", recognizedEmployee.getUsername() + " checked in successfully for " + checkType);
//                    break;
//                case "late":
//                    response.put("message", recognizedEmployee.getUsername() + " is late for " + checkType);
//                    break;
//                case "absent":
//                    response.put("message", recognizedEmployee.getUsername() + " is marked absent for halfday");
//                    break;
//            }
//
//        } catch (Exception e) {
//            response.put("status", "error");
//            response.put("message", "Face recognition processing failed: " + e.getMessage());
//        }
//
//        return response;
//    }
//
//    private Employee performFaceRecognition(MultipartFile image) {
//        try {
//            // Convert image to byte array
//            byte[] imageBytes = image.getBytes();
//
//            // Get all employees with profile images
//            List<Employee> employees = employeeRepository.findAll();
//
//            // Simple face matching logic (replace with actual face recognition library)
//            for (Employee employee : employees) {
//                if (employee.getProfileImage() != null) {
//                    // In a real implementation, you would use a face recognition library
//                    // like face_recognition (Python), OpenCV, or cloud services like AWS Rekognition
//
//                    // For demo purposes, we'll simulate recognition based on image size similarity
//                    if (Math.abs(imageBytes.length - employee.getProfileImage().length) < 1000) {
//                        return employee;
//                    }
//                }
//            }
//
//            // If no match found, return the first employee for demo purposes
//            // In production, this should return null
//            if (!employees.isEmpty()) {
//                return employees.get(0); // Demo: return first employee
//            }
//
//            return null;
//
//        } catch (Exception e) {
//            throw new RuntimeException("Face recognition failed: " + e.getMessage());
//        }
//    }
//
//    private String validateTimeAndUpdateStatus(Attendance attendance, String checkType, LocalTime currentTime) {
//
//        if ("permission".equals(checkType)) {
//            return validatePermissionTime(attendance, currentTime);
//        } else if ("halfday".equals(checkType)) {
//            return validateHalfdayTime(attendance, currentTime);
//        }
//
//        return "error";
//    }
//
//    private String validatePermissionTime(Attendance attendance, LocalTime currentTime) {
//        LocalTime permissionFrom = attendance.getPermissionFrom();
//        LocalTime permissionTo = attendance.getPermissionTo();
//
//        if (permissionFrom == null || permissionTo == null) {
//            return "no_permission";
//        }
//
//        // Check if current time is within permission time range
//        if (currentTime.isAfter(permissionFrom) && currentTime.isBefore(permissionTo)) {
//            // Within time - mark as permission
//            attendance.setStatus("permission");
//            attendance.setCheckinTime(currentTime);
//            return "success";
//        } else {
//            // Late for permission
//            attendance.setStatus("late");
//            attendance.setCheckinTime(currentTime);
//            return "late";
//        }
//    }
//
//    private String validateHalfdayTime(Attendance attendance, LocalTime currentTime) {
//        LocalTime halfdayFrom = attendance.getHalfdayFrom();
//        LocalTime halfdayTo = attendance.getHalfdayTo();
//
//        if (halfdayFrom == null || halfdayTo == null) {
//            return "no_permission";
//        }
//
//        // Check if current time is within halfday time range
//        if (currentTime.isAfter(halfdayFrom) && currentTime.isBefore(halfdayTo)) {
//            // Within time - mark as halfday
//            attendance.setStatus("halfday");
//            attendance.setCheckinTime(currentTime);
//            return "success";
//        } else {
//            // Outside halfday time - mark as absent
//            attendance.setStatus("absent");
//            attendance.setCheckinTime(currentTime);
//            return "absent";
//        }
//    }
//
//    private Map<String, Object> createEmployeeResponse(Employee employee) {
//        Map<String, Object> employeeData = new HashMap<>();
//        employeeData.put("id", employee.getId());
//        employeeData.put("username", employee.getUsername());
//        employeeData.put("department", employee.getDepartment());
//        employeeData.put("designation", employee.getDesignation());
//        employeeData.put("gender", employee.getGender());
//        employeeData.put("email", employee.getEmail());
//
//        // Convert profile image to base64 if exists
//        if (employee.getProfileImage() != null) {
//            employeeData.put("profileImage", Base64.getEncoder().encodeToString(employee.getProfileImage()));
//        }
//
//        return employeeData;
//    }
//
//    private Map<String, Object> createAttendanceResponse(Attendance attendance) {
//        Map<String, Object> attendanceData = new HashMap<>();
//        attendanceData.put("id", attendance.getId());
//        attendanceData.put("employeeId", attendance.getEmployeeId());
//        attendanceData.put("employeeName", attendance.getEmployeeName());
//        attendanceData.put("date", attendance.getDate());
//        attendanceData.put("checkinTime", attendance.getCheckinTime());
//        attendanceData.put("status", attendance.getStatus());
//        attendanceData.put("permissionFrom", attendance.getPermissionFrom());
//        attendanceData.put("permissionTo", attendance.getPermissionTo());
//        attendanceData.put("halfdayFrom", attendance.getHalfdayFrom());
//        attendanceData.put("halfdayTo", attendance.getHalfdayTo());
//        attendanceData.put("halfdayType", attendance.getHalfdayType());
//        attendanceData.put("reason", attendance.getReason());
//
//        return attendanceData;
//    }
//
//    private Map<String, Object> createTimeInfo(LocalTime currentTime) {
//        Map<String, Object> timeInfo = new HashMap<>();
//        timeInfo.put("currentTime", currentTime.toString());
//        timeInfo.put("currentDate", LocalDate.now().toString());
//        timeInfo.put("timestamp", System.currentTimeMillis());
//
//        return timeInfo;
//    }
//
//    public Map<String, Object> getEmployeePermissions(String employeeId, String date) {
//        LocalDate searchDate = (date != null) ? LocalDate.parse(date) : LocalDate.now();
//
//        Attendance attendance = permissionHalfdayRepository.findByEmployeeIdAndDate(employeeId, searchDate);
//
//        Map<String, Object> response = new HashMap<>();
//
//        if (attendance != null) {
//            response.put("status", "success");
//            response.put("attendance", createAttendanceResponse(attendance));
//        } else {
//            response.put("status", "not_found");
//            response.put("message", "No permissions found for the specified date");
//        }
//
//        return response;
//    }
//}