//package com.Attendance.BackEnd_Attendance.Controller;
//
//import com.Attendance.BackEnd_Attendance.Service.PermissionHalfdayService;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.http.ResponseEntity;
//import org.springframework.web.bind.annotation.*;
//import org.springframework.web.multipart.MultipartFile;
//
//import java.util.HashMap;
//import java.util.Map;
//
//@RestController
//@RequestMapping("/api/permissionhalfday")
//@CrossOrigin(origins = "*")
//public class PermissionHalfdayController {
//
//    @Autowired
//    private PermissionHalfdayService permissionHalfdayService;
//
//    @PostMapping("/face-recognition")
//    public ResponseEntity<Map<String, Object>> faceRecognition(
//            @RequestParam("image") MultipartFile image,
//            @RequestParam("checkType") String checkType,
//            @RequestParam("timestamp") String timestamp,
//            @RequestParam("currentTime") String currentTime) {
//
//        Map<String, Object> response = new HashMap<>();
//
//        try {
//            // Validate input parameters
//            if (image == null || image.isEmpty()) {
//                response.put("status", "error");
//                response.put("message", "Image is required");
//                return ResponseEntity.badRequest().body(response);
//            }
//
//            if (checkType == null || (!checkType.equals("permission") && !checkType.equals("halfday"))) {
//                response.put("status", "error");
//                response.put("message", "Invalid check type. Must be 'permission' or 'halfday'");
//                return ResponseEntity.badRequest().body(response);
//            }
//
//            // Process face recognition and attendance check
//            Map<String, Object> result = permissionHalfdayService.processFaceRecognition(
//                    image, checkType, timestamp, currentTime
//            );
//
//            return ResponseEntity.ok(result);
//
//        } catch (Exception e) {
//            response.put("status", "error");
//            response.put("message", "Face recognition failed: " + e.getMessage());
//            return ResponseEntity.internalServerError().body(response);
//        }
//    }
//
//    @GetMapping("/employee/{employeeId}/permissions")
//    public ResponseEntity<Map<String, Object>> getEmployeePermissions(
//            @PathVariable String employeeId,
//            @RequestParam(required = false) String date) {
//
//        try {
//            Map<String, Object> permissions = permissionHalfdayService.getEmployeePermissions(employeeId, date);
//            return ResponseEntity.ok(permissions);
//        } catch (Exception e) {
//            Map<String, Object> response = new HashMap<>();
//            response.put("status", "error");
//            response.put("message", "Failed to get permissions: " + e.getMessage());
//            return ResponseEntity.internalServerError().body(response);
//        }
//    }
//
//    @GetMapping("/test")
//    public ResponseEntity<Map<String, Object>> testEndpoint() {
//        Map<String, Object> response = new HashMap<>();
//        response.put("status", "success");
//        response.put("message", "Permission Halfday API is working");
//        response.put("timestamp", System.currentTimeMillis());
//        return ResponseEntity.ok(response);
//    }
//}