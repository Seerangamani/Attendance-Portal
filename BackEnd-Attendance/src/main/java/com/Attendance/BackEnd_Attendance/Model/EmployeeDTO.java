package com.Attendance.BackEnd_Attendance.Model;

public class EmployeeDTO {
    private String id;
    private String usertype;
    private String department;
    private String designation;
    private String username;
    private String gender;
    private String email;
    private String password;
    private String profileImageBase64; // For API response

    // Default constructor
    public EmployeeDTO() {
    }

    // Constructor from Employee entity
    public EmployeeDTO(String id, String usertype, String department, String designation,
                       String username, String gender, String email, String password, String profileImageBase64) {
        this.id = id;
        this.usertype = usertype;
        this.department = department;
        this.designation = designation;
        this.username = username;
        this.gender = gender;
        this.email = email;
        this.password = password;
        this.profileImageBase64 = profileImageBase64;
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getUsertype() {
        return usertype;
    }

    public void setUsertype(String usertype) {
        this.usertype = usertype;
    }

    public String getDepartment() {
        return department;
    }

    public void setDepartment(String department) {
        this.department = department;
    }

    public String getDesignation() {
        return designation;
    }

    public void setDesignation(String designation) {
        this.designation = designation;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getProfileImageBase64() {
        return profileImageBase64;
    }

    public void setProfileImageBase64(String profileImageBase64) {
        this.profileImageBase64 = profileImageBase64;
    }
}