package com.Attendance.BackEnd_Attendance.Controller;

import com.Attendance.BackEnd_Attendance.Model.EmployeeDTO;
import com.Attendance.BackEnd_Attendance.Model.Employee;
import com.Attendance.BackEnd_Attendance.Repository.EmployeeRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/employees")
@CrossOrigin(origins = "*", allowedHeaders = "*")
public class EmployeeController {

    @Autowired
    private EmployeeRepository employeeRepository;

    @PostMapping(value = "/add", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @CrossOrigin(origins = "*", allowedHeaders = "*")
    public ResponseEntity<String> addEmployee(
            @RequestParam("id") String id,
            @RequestParam("usertype") String userType,
            @RequestParam("department") String department,
            @RequestParam("designation") String designation,
            @RequestParam("username") String username,
            @RequestParam("gender") String gender,
            @RequestParam("email") String email,
            @RequestParam("password") String password,
            @RequestParam("profileImage") MultipartFile profileImage) {

        try {
            Employee employee = new Employee();
            employee.setId(id);
            employee.setUsertype(userType);
            employee.setDepartment(department);
            employee.setDesignation(designation);
            employee.setUsername(username);
            employee.setGender(gender);
            employee.setEmail(email);
            employee.setPassword(password);
            employee.setProfileImage(profileImage.getBytes());

            employeeRepository.save(employee);
            return ResponseEntity.ok("Employee added successfully!");

        } catch (IOException e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error saving employee or uploading image.");
        }
    }

    @PutMapping(value = "/update/{id}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @CrossOrigin(origins = "*", allowedHeaders = "*")
    public ResponseEntity<String> updateEmployeeWithImage(
            @PathVariable String id,
            @RequestParam("usertype") String userType,
            @RequestParam("department") String department,
            @RequestParam("designation") String designation,
            @RequestParam("username") String username,
            @RequestParam("gender") String gender,
            @RequestParam("email") String email,
            @RequestParam("password") String password,
            @RequestParam(value = "profileImage", required = false) MultipartFile profileImage) {

        Optional<Employee> optionalEmployee = employeeRepository.findById(id);
        if (!optionalEmployee.isPresent()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body("Employee with ID " + id + " not found.");
        }

        try {
            Employee employee = optionalEmployee.get();
            employee.setUsertype(userType);
            employee.setDepartment(department);
            employee.setDesignation(designation);
            employee.setUsername(username);
            employee.setGender(gender);
            employee.setEmail(email);
            employee.setPassword(password);

            if (profileImage != null && !profileImage.isEmpty()) {
                employee.setProfileImage(profileImage.getBytes());
            }

            employeeRepository.save(employee);
            return ResponseEntity.ok("Employee updated successfully.");

        } catch (IOException e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error processing the profile image.");
        }
    }

    @GetMapping
    @CrossOrigin(origins = "*", allowedHeaders = "*")
    public List<EmployeeDTO> getAllEmployees() {
        List<Employee> employees = employeeRepository.findAll();
        System.out.println("Total employees found: " + employees.size());

        return employees.stream().map(this::convertToDTO).collect(Collectors.toList());
    }

    @GetMapping("/{id}")
    @CrossOrigin(origins = "*", allowedHeaders = "*")
    public ResponseEntity<EmployeeDTO> getEmployeeById(@PathVariable String id) {
        Optional<Employee> employee = employeeRepository.findById(id);
        if (employee.isPresent()) {
            return ResponseEntity.ok(convertToDTO(employee.get()));
        } else {
            return ResponseEntity.notFound().build();
        }
    }

    @DeleteMapping("/{id}")
    @CrossOrigin(origins = "*", allowedHeaders = "*")
    public ResponseEntity<String> deleteEmployee(@PathVariable String id) {
        if (!employeeRepository.existsById(id)) {
            return ResponseEntity.badRequest().body("Employee not found with ID: " + id);
        }
        employeeRepository.deleteById(id);
        return ResponseEntity.ok("Employee deleted successfully.");
    }

    @GetMapping("/face-image/{id}")
    @CrossOrigin(origins = "*", allowedHeaders = "*")
    public ResponseEntity<?> getEmployeeFaceImage(@PathVariable String id) {
        return employeeRepository.findById(id)
                .map(employee -> {
                    byte[] imageBytes = employee.getProfileImage();
                    if (imageBytes == null || imageBytes.length == 0) {
                        return ResponseEntity.badRequest()
                                .body("No profile image available for employee ID: " + id);
                    }
                    String base64Image = Base64.getEncoder().encodeToString(imageBytes);
                    Map<String, String> response = new HashMap<>();
                    response.put("base64Image", base64Image);
                    return ResponseEntity.ok(response);
                })
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/face-image-bytes/{id}")
    @CrossOrigin(origins = "*", allowedHeaders = "*")
    public ResponseEntity<byte[]> getEmployeeFaceImageBytes(@PathVariable String id) {
        return employeeRepository.findById(id)
                .filter(employee -> employee.getProfileImage() != null && employee.getProfileImage().length > 0)
                .map(employee -> {
                    HttpHeaders headers = new HttpHeaders();
                    headers.setContentType(MediaType.IMAGE_JPEG);
                    return new ResponseEntity<>(employee.getProfileImage(), headers, HttpStatus.OK);
                })
                .orElse(ResponseEntity.notFound().build());
    }

    // Helper method to convert Employee to EmployeeDTO
    private EmployeeDTO convertToDTO(Employee employee) {
        String base64Image = null;
        if (employee.getProfileImage() != null && employee.getProfileImage().length > 0) {
            base64Image = Base64.getEncoder().encodeToString(employee.getProfileImage());
        }

        return new EmployeeDTO(
                employee.getId(),
                employee.getUsertype(),
                employee.getDepartment(),
                employee.getDesignation(),
                employee.getUsername(),
                employee.getGender(),
                employee.getEmail(),
                employee.getPassword(),
                base64Image
        );
    }
}