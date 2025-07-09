package com.Attendance.BackEnd_Attendance.Controller;

import com.Attendance.BackEnd_Attendance.Model.Admin;
import com.Attendance.BackEnd_Attendance.Model.Employee;
import com.Attendance.BackEnd_Attendance.Model.AttendanceUser;
import com.Attendance.BackEnd_Attendance.Service.LoginService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Base64;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:3000") // adjust if needed
public class LoginController {

    @Autowired
    private LoginService loginService;

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody Map<String, String> request) {
        try {
            System.out.println("Login attempt received with data: " + request);
            String id = request.get("id");
            String email = request.get("email");
            String password = request.get("password");
            String usertype = request.get("usertype");

            // Validate required fields
            if (id == null || id.trim().isEmpty() ||
                    email == null || email.trim().isEmpty() ||
                    password == null || password.trim().isEmpty() ||
                    usertype == null || usertype.trim().isEmpty()) {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body
                        (Map.of("Error:","All Fields Are Required ."));

            }

            if ("admin".equalsIgnoreCase(usertype)) {
                Optional<Admin> adminOpt = loginService.adminLogin(id, email, password);
                if (adminOpt.isPresent()) {
                    Admin admin = adminOpt.get();
                    return ResponseEntity.ok(Map.of(
                            "usertype", "admin",
                            "id", admin.getId(),
                            "email", admin.getEmail(),
                            "message", "Admin login successful"
                    ));
                } else {
                    return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(
                            Map.of("error", "Invalid admin credentials")
                    );
                }
            } else if ("public".equalsIgnoreCase(usertype)) {
                Optional<AttendanceUser> publicUserOpt = loginService.publicLogin(id, password);
                if (publicUserOpt.isPresent()) {
                    AttendanceUser pub = publicUserOpt.get();
                    return ResponseEntity.ok(Map.of(
                            "usertype", "public",
                            "id", pub.getId(),
                            "username", pub.getUsername(),
                            "email", email, // Pass through the provided email
                            "message", "Public login successful"
                    ));
                } else {
                    return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(
                            Map.of("error", "Invalid public user credentials")
                    );
                }
            } else if ("employee".equalsIgnoreCase(usertype)) {
                // Modified employee login - now only checks ID, email, and password
                Optional<Employee> empOpt = loginService.employeeLogin(id, email, password);
                if (empOpt.isPresent()) {
                    Employee emp = empOpt.get();

                    // Return employee data as a Map instead of DTO
                    return ResponseEntity.ok(Map.of(
                            "usertype", "employee",
                            "id", emp.getId(),
                            "username", emp.getUsername() != null ? emp.getUsername() : "",
                            "email", emp.getEmail(),
                            "department", emp.getDepartment() != null ? emp.getDepartment() : "",
                            "designation", emp.getDesignation() != null ? emp.getDesignation() : "",
                            "gender", emp.getGender() != null ? emp.getGender() : "",
                            "actualUserType", emp.getUsertype() != null ? emp.getUsertype() : "", // Add actual usertype
                            "profileImage", emp.getProfileImage() != null ? Base64.getEncoder().encodeToString(emp.getProfileImage()) : null,
                            "message", "Employee login successful"
                    ));
                } else {
                    return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(
                            Map.of("error", "Invalid employee credentials. Please check your ID, email, and password.")
                    );
                }
            } else {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(
                        Map.of("error", "Invalid user type. Please select Admin, Employee, or Public.")
                );
            }
        } catch (Exception e) {
            e.printStackTrace(); // log error
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
                    Map.of("error", "Internal Server Error: " + e.getMessage())
            );
        }
    }

    // New endpoint to get individual employee data
    @GetMapping("/employee/{id}")
    public ResponseEntity<?> getEmployee(@PathVariable String id) {
        try {
            System.out.println("Getting employee data for ID: " + id);

            Optional<Employee> empOpt = loginService.getEmployeeById(id);
            if (empOpt.isPresent()) {
                Employee emp = empOpt.get();

                return ResponseEntity.ok(Map.of(
                        "id", emp.getId(),
                        "username", emp.getUsername() != null ? emp.getUsername() : "",
                        "email", emp.getEmail(),
                        "department", emp.getDepartment() != null ? emp.getDepartment() : "",
                        "designation", emp.getDesignation() != null ? emp.getDesignation() : "",
                        "gender", emp.getGender() != null ? emp.getGender() : "",
                        "usertype", emp.getUsertype() != null ? emp.getUsertype() : "",
                        "profileImage", emp.getProfileImage() != null ? Base64.getEncoder().encodeToString(emp.getProfileImage()) : null
                ));
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(
                        Map.of("error", "Employee not found")
                );
            }
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
                    Map.of("error", "Internal Server Error: " + e.getMessage())
            );
        }
    }
}