package com.Attendance.BackEnd_Attendance.Repository;

import com.Attendance.BackEnd_Attendance.Model.MonthlyPermissionTracker;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface MonthlyPermissionTrackerRepository extends JpaRepository<MonthlyPermissionTracker, Long> {
    Optional<MonthlyPermissionTracker> findByEmployeeIdAndYearMonth(String employeeId, String yearMonth);
    List<MonthlyPermissionTracker> findByEmployeeId(String employeeId);
    List<MonthlyPermissionTracker> findByYearMonth(String yearMonth);
}