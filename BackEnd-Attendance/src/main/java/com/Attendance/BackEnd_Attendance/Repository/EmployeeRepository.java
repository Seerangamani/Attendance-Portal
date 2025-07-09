package com.Attendance.BackEnd_Attendance.Repository;

import com.Attendance.BackEnd_Attendance.Model.Employee;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface EmployeeRepository extends JpaRepository<Employee, String> {

    // Updated method - removed usertype parameter
    Optional<Employee> findByIdAndEmailAndPassword(String id, String email, String password);

    // Keep the old method for backward compatibility if needed elsewhere
    Optional<Employee> findByIdAndEmailAndPasswordAndUsertype(String id, String email, String password, String usertype);

    Optional<Employee> findByUsername(String username);

    List<Employee> findByDepartment(String department);

    // This method is inherited from JpaRepository, but explicitly declaring for clarity
    Optional<Employee> findById(String id);
}