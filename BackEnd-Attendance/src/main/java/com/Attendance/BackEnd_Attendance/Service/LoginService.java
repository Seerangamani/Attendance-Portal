package com.Attendance.BackEnd_Attendance.Service;

import com.Attendance.BackEnd_Attendance.Model.Admin;
import com.Attendance.BackEnd_Attendance.Model.Employee;
import com.Attendance.BackEnd_Attendance.Model.AttendanceUser;
import com.Attendance.BackEnd_Attendance.Repository.AdminRepository;
import com.Attendance.BackEnd_Attendance.Repository.EmployeeRepository;
import com.Attendance.BackEnd_Attendance.Repository.AttendanceUserRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class LoginService {

    @Autowired
    AdminRepository adminRepository;

    @Autowired
    EmployeeRepository employeeRepository;

    @Autowired
    AttendanceUserRepository attendanceUserRepository;

    public Optional<Admin> adminLogin(String id, String email, String password) {
        return adminRepository.findByIdAndEmailAndPassword(id, email, password);
    }

    // Updated employee login - removed usertype parameter
    public Optional<Employee> employeeLogin(String id, String email, String password) {
        return employeeRepository.findByIdAndEmailAndPassword(id, email, password);
    }

    public Optional<AttendanceUser> publicLogin(String id, String password) {
        return attendanceUserRepository.findByIdAndPassword(id, password);
    }

    // New method to get employee by ID
    public Optional<Employee> getEmployeeById(String id) {
        return employeeRepository.findById(id);
    }
}