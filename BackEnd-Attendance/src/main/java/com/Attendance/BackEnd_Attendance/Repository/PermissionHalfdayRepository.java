//package com.Attendance.BackEnd_Attendance.Repository;
//
//import com.Attendance.BackEnd_Attendance.Model.Attendance;
//import org.springframework.data.jpa.repository.JpaRepository;
//import org.springframework.data.jpa.repository.Query;
//import org.springframework.data.repository.query.Param;
//import org.springframework.stereotype.Repository;
//
//import java.time.LocalDate;
//import java.time.LocalTime;
//import java.util.List;
//import java.util.Optional;
//
//@Repository
//public interface PermissionHalfdayRepository extends JpaRepository<Attendance, Long> {
//
//    /**
//     * Find attendance record by employee ID and date
//     * This is the main method used in your service for face recognition
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date")
//    Attendance findByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);
//
//    /**
//     * Find permission allocation for employee on specific date
//     * Used to check if employee has permission allocated for today
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date " +
//            "AND (a.permissionFrom IS NOT NULL AND a.permissionTo IS NOT NULL)")
//    Optional<Attendance> findPermissionByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);
//
//    /**
//     * Find halfday allocation for employee on specific date
//     * Used to check if employee has halfday allocated for today
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date " +
//            "AND (a.halfdayFrom IS NOT NULL AND a.halfdayTo IS NOT NULL)")
//    Optional<Attendance> findHalfdayByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);
//
//    /**
//     * Check if employee has valid permission for current time
//     * Used to validate if employee comes within permission time
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date " +
//            "AND a.permissionFrom <= :currentTime AND a.permissionTo >= :currentTime")
//    Optional<Attendance> findValidPermissionForTime(@Param("employeeId") String employeeId,
//                                                    @Param("date") LocalDate date,
//                                                    @Param("currentTime") LocalTime currentTime);
//
//    /**
//     * Check if employee has valid halfday for current time
//     * Used to validate if employee comes within halfday time
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date " +
//            "AND a.halfdayFrom <= :currentTime AND a.halfdayTo >= :currentTime")
//    Optional<Attendance> findValidHalfdayForTime(@Param("employeeId") String employeeId,
//                                                 @Param("date") LocalDate date,
//                                                 @Param("currentTime") LocalTime currentTime);
//
//    /**
//     * Find all permission records for a specific date
//     * Used for admin dashboard to see all permissions for the day
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.permissionFrom IS NOT NULL")
//    List<Attendance> findAllPermissionsForDate(@Param("date") LocalDate date);
//
//    /**
//     * Find all halfday records for a specific date
//     * Used for admin dashboard to see all halfdays for the day
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.halfdayFrom IS NOT NULL")
//    List<Attendance> findAllHalfdaysForDate(@Param("date") LocalDate date);
//
//    /**
//     * Find permission/halfday records by status
//     * Used to filter by status (permission, halfday, late, absent)
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.status = :status AND a.date = :date")
//    List<Attendance> findByStatusAndDate(@Param("status") String status, @Param("date") LocalDate date);
//
//    /**
//     * Find late permission users for a specific date
//     * Used to get employees who came late for permission
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.status = 'late' " +
//            "AND a.permissionFrom IS NOT NULL")
//    List<Attendance> findLatePermissionUsers(@Param("date") LocalDate date);
//
//    /**
//     * Find absent halfday users for a specific date
//     * Used to get employees who are marked absent for halfday
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.status = 'absent' " +
//            "AND a.halfdayFrom IS NOT NULL")
//    List<Attendance> findAbsentHalfdayUsers(@Param("date") LocalDate date);
//
//    /**
//     * Find attendance records with check-in time for specific date
//     * Used to get employees who have successfully checked in
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.checkinTime IS NOT NULL")
//    List<Attendance> findCheckedInRecords(@Param("date") LocalDate date);
//
//    /**
//     * Find pending attendance records (no check-in time yet)
//     * Used to get employees who haven't checked in yet
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.checkinTime IS NULL " +
//            "AND (a.permissionFrom IS NOT NULL OR a.halfdayFrom IS NOT NULL)")
//    List<Attendance> findPendingPermissionHalfdayRecords(@Param("date") LocalDate date);
//
//    /**
//     * Check if employee has any permission/halfday allocation for date
//     * Used to validate if employee has any allocation before face recognition
//     */
//    @Query("SELECT CASE WHEN COUNT(a) > 0 THEN true ELSE false END FROM Attendance a " +
//            "WHERE a.employeeId = :employeeId AND a.date = :date " +
//            "AND (a.permissionFrom IS NOT NULL OR a.halfdayFrom IS NOT NULL)")
//    boolean hasPermissionOrHalfdayAllocation(@Param("employeeId") String employeeId, @Param("date") LocalDate date);
//
//    /**
//     * Find attendance records by employee name and date
//     * Used for searching by employee name
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.employeeName = :employeeName AND a.date = :date")
//    Optional<Attendance> findByEmployeeNameAndDate(@Param("employeeName") String employeeName, @Param("date") LocalDate date);
//
//    /**
//     * Find attendance records by halfday type
//     * Used to filter by halfday type (morning, afternoon, etc.)
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.halfdayType = :halfdayType AND a.date = :date")
//    List<Attendance> findByHalfdayTypeAndDate(@Param("halfdayType") String halfdayType, @Param("date") LocalDate date);
//
//    /**
//     * Find attendance records with specific reason
//     * Used to filter by reason for permission/halfday
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.reason = :reason AND a.date = :date")
//    List<Attendance> findByReasonAndDate(@Param("reason") String reason, @Param("date") LocalDate date);
//
//    /**
//     * Get attendance summary for date
//     * Used for dashboard statistics
//     */
//    @Query("SELECT a.status, COUNT(a) FROM Attendance a WHERE a.date = :date " +
//            "AND (a.permissionFrom IS NOT NULL OR a.halfdayFrom IS NOT NULL) GROUP BY a.status")
//    List<Object[]> getPermissionHalfdayStatistics(@Param("date") LocalDate date);
//
//    /**
//     * Find attendance records by date range
//     * Used for reports and analytics
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date BETWEEN :startDate AND :endDate " +
//            "AND (a.permissionFrom IS NOT NULL OR a.halfdayFrom IS NOT NULL) ORDER BY a.date DESC")
//    List<Attendance> findByDateRange(@Param("startDate") LocalDate startDate, @Param("endDate") LocalDate endDate);
//
//    /**
//     * Find employees who came within time for permission
//     * Used for positive reporting
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.status = 'permission' " +
//            "AND a.permissionFrom IS NOT NULL AND a.checkinTime IS NOT NULL")
//    List<Attendance> findSuccessfulPermissionUsers(@Param("date") LocalDate date);
//
//    /**
//     * Find employees who came within time for halfday
//     * Used for positive reporting
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.date = :date AND a.status = 'halfday' " +
//            "AND a.halfdayFrom IS NOT NULL AND a.checkinTime IS NOT NULL")
//    List<Attendance> findSuccessfulHalfdayUsers(@Param("date") LocalDate date);
//
//    /**
//     * Delete attendance record by employee ID and date
//     * Used for cleanup or correction
//     */
//    @Query("DELETE FROM Attendance a WHERE a.employeeId = :employeeId AND a.date = :date")
//    void deleteByEmployeeIdAndDate(@Param("employeeId") String employeeId, @Param("date") LocalDate date);
//
//    /**
//     * Count total permission allocations for date
//     * Used for dashboard metrics
//     */
//    @Query("SELECT COUNT(a) FROM Attendance a WHERE a.date = :date AND a.permissionFrom IS NOT NULL")
//    Long countPermissionAllocations(@Param("date") LocalDate date);
//
//    /**
//     * Count total halfday allocations for date
//     * Used for dashboard metrics
//     */
//    @Query("SELECT COUNT(a) FROM Attendance a WHERE a.date = :date AND a.halfdayFrom IS NOT NULL")
//    Long countHalfdayAllocations(@Param("date") LocalDate date);
//
//    /**
//     * Find latest attendance record for employee
//     * Used to get most recent attendance record
//     */
//    @Query("SELECT a FROM Attendance a WHERE a.employeeId = :employeeId " +
//            "ORDER BY a.date DESC, a.checkinTime DESC LIMIT 1")
//    Optional<Attendance> findLatestAttendanceByEmployeeId(@Param("employeeId") String employeeId);
//}