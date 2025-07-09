package com.Attendance.BackEnd_Attendance.Repository;

import com.Attendance.BackEnd_Attendance.Model.Admin;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;


import java.util.Optional;
@Repository
public interface AdminRepository extends JpaRepository<Admin, String> {
    Optional<Admin> findByIdAndEmailAndPassword(String id, String email, String password);

}
