package com.Attendance.BackEnd_Attendance.Repository;

import com.Attendance.BackEnd_Attendance.Model.AttendanceUser;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface AttendanceUserRepository extends JpaRepository<AttendanceUser, String> {
    Optional<AttendanceUser> findByIdAndPassword(String id, String password);
}
