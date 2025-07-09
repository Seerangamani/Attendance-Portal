//package com.Attendance.BackEnd_Attendance.Service;
//
//import com.Attendance.BackEnd_Attendance.Model.Attendance;
//import com.Attendance.BackEnd_Attendance.Model.Employee;
//import com.Attendance.BackEnd_Attendance.Model.AttendanceStatsDTO;
//import com.Attendance.BackEnd_Attendance.Repository.AttendanceConditionRepository;
//import com.Attendance.BackEnd_Attendance.Repository.EmployeeRepository;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.stereotype.Service;
//
//import java.time.DayOfWeek;
//import java.time.LocalDate;
//import java.time.LocalTime;
//import java.time.format.DateTimeFormatter;
//import java.time.temporal.TemporalAdjusters;
//import java.util.*;
//import java.util.stream.Collectors;
//
//@Service
//public class AttendanceConditionService {
//
//    @Autowired
//    private AttendanceConditionRepository attendanceConditionRepository;
//
//    @Autowired
//    private EmployeeRepository employeeRepository;
//
//    /**
//     * Get attendance statistics for a specific employee
//     */
//    public AttendanceStatsDTO getAttendanceStatistics(String employeeId) {
//        List<Attendance> allAttendance = attendanceConditionRepository.findByEmployeeId(employeeId);
//
//        AttendanceStatsDTO stats = new AttendanceStatsDTO();
//        stats.setEmployeeId(employeeId);
//        stats.setTotal(allAttendance.size());
//
//        Map<String, Long> statusCounts = allAttendance.stream()
//                .collect(Collectors.groupingBy(
//                        attendance -> attendance.getStatus() != null ? attendance.getStatus().toLowerCase() : "absent",
//                        Collectors.counting()
//                ));
//
//        stats.setPresent(statusCounts.getOrDefault("present", 0L).intValue());
//        stats.setAbsent(statusCounts.getOrDefault("absent", 0L).intValue());
//        stats.setLate(statusCounts.getOrDefault("late", 0L).intValue());
//        stats.setOnduty(statusCounts.getOrDefault("onduty", 0L).intValue());
//        stats.setHalfday(statusCounts.getOrDefault("halfday", 0L).intValue());
//        stats.setPermission(statusCounts.getOrDefault("permission", 0L).intValue());
//
//        return stats;
//    }
//
//    /**
//     * Get today's attendance for an employee
//     */
//    public Attendance getTodayAttendance(String employeeId) {
//        LocalDate today = LocalDate.now();
//        return attendanceConditionRepository.findByEmployeeIdAndDate(employeeId, today);
//    }
//
//    /**
//     * Get filtered attendance records based on filter type
//     */
//    public List<Attendance> getFilteredAttendance(String employeeId, String filter, LocalDate startDate, LocalDate endDate) {
//        if (startDate != null && endDate != null) {
//            return attendanceConditionRepository.findByEmployeeIdAndDateBetween(employeeId, startDate, endDate);
//        }
//
//        if (filter == null) {
//            filter = "today";
//        }
//
//        switch (filter.toLowerCase()) {
//            case "today":
//                return getTodayAttendanceAsList(employeeId);
//            case "weekly":
//                return getWeeklyAttendance(employeeId);
//            case "monthly":
//                return getMonthlyAttendance(employeeId);
//            case "overall":
//                return attendanceConditionRepository.findByEmployeeIdOrderByDateDesc(employeeId);
//            default:
//                return getTodayAttendanceAsList(employeeId);
//        }
//    }
//
//    /**
//     * Get attendance records by status
//     */
//    public List<Attendance> getAttendanceByStatus(String employeeId, String status) {
//        return attendanceConditionRepository.findByEmployeeIdAndStatus(employeeId, status);
//    }
//
//    /**
//     * Get attendance records for date range
//     */
//    public List<Attendance> getAttendanceByDateRange(String employeeId, LocalDate startDate, LocalDate endDate) {
//        return attendanceConditionRepository.findByEmployeeIdAndDateBetween(employeeId, startDate, endDate);
//    }
//
//    /**
//     * Get monthly attendance summary
//     */
//    public Map<String, Integer> getMonthlyAttendanceSummary(String employeeId, int year, int month) {
//        LocalDate startDate = LocalDate.of(year, month, 1);
//        LocalDate endDate = startDate.with(TemporalAdjusters.lastDayOfMonth());
//
//        List<Attendance> monthlyAttendance = attendanceConditionRepository
//                .findByEmployeeIdAndDateBetween(employeeId, startDate, endDate);
//
//        return calculateStatusSummary(monthlyAttendance);
//    }
//
//    /**
//     * Get weekly attendance summary
//     */
//    public Map<String, Integer> getWeeklyAttendanceSummary(String employeeId, LocalDate weekStartDate) {
//        LocalDate weekEndDate = weekStartDate.plusDays(6);
//
//        List<Attendance> weeklyAttendance = attendanceConditionRepository
//                .findByEmployeeIdAndDateBetween(employeeId, weekStartDate, weekEndDate);
//
//        return calculateStatusSummary(weeklyAttendance);
//    }
//
//    /**
//     * Check if employee is present today
//     */
//    public boolean isEmployeePresentToday(String employeeId) {
//        LocalDate today = LocalDate.now();
//        Attendance todayAttendance = attendanceConditionRepository.findByEmployeeIdAndDate(employeeId, today);
//
//        if (todayAttendance == null) {
//            return false;
//        }
//
//        String status = todayAttendance.getStatus();
//        return "present".equalsIgnoreCase(status) ||
//                "late".equalsIgnoreCase(status) ||
//                "onduty".equalsIgnoreCase(status) ||
//                "halfday".equalsIgnoreCase(status) ||
//                "permission".equalsIgnoreCase(status);
//    }
//
//    /**
//     * Get employees with specific attendance status for today
//     */
//    public List<Attendance> getEmployeesByTodayStatus(String status) {
//        LocalDate today = LocalDate.now();
//        return attendanceConditionRepository.findByDateAndStatus(today, status);
//    }
//
//    /**
//     * Get attendance statistics for all employees
//     */
//    public Map<String, AttendanceStatsDTO> getAllEmployeesAttendanceStats() {
//        List<Employee> allEmployees = employeeRepository.findAll();
//        Map<String, AttendanceStatsDTO> allStats = new HashMap<>();
//
//        for (Employee employee : allEmployees) {
//            AttendanceStatsDTO stats = getAttendanceStatistics(employee.getId());
//            allStats.put(employee.getId(), stats);
//        }
//
//        return allStats;
//    }
//
//    /**
//     * Get late arrivals for a specific date range
//     */
//    public List<Attendance> getLateArrivals(LocalDate startDate, LocalDate endDate) {
//        return attendanceConditionRepository.findByDateBetweenAndStatus(startDate, endDate, "late");
//    }
//
//    /**
//     * Get employees with perfect attendance for a month
//     */
//    public List<String> getEmployeesWithPerfectAttendance(int year, int month) {
//        LocalDate startDate = LocalDate.of(year, month, 1);
//        LocalDate endDate = startDate.with(TemporalAdjusters.lastDayOfMonth());
//
//        List<Employee> allEmployees = employeeRepository.findAll();
//        List<String> perfectAttendanceEmployees = new ArrayList<>();
//
//        for (Employee employee : allEmployees) {
//            List<Attendance> monthlyAttendance = attendanceConditionRepository
//                    .findByEmployeeIdAndDateBetween(employee.getId(), startDate, endDate);
//
//            // Calculate working days in the month (excluding weekends)
//            long workingDays = startDate.datesUntil(endDate.plusDays(1))
//                    .filter(date -> date.getDayOfWeek() != DayOfWeek.SATURDAY &&
//                            date.getDayOfWeek() != DayOfWeek.SUNDAY)
//                    .count();
//
//            long presentDays = monthlyAttendance.stream()
//                    .filter(att -> "present".equalsIgnoreCase(att.getStatus()) ||
//                            "late".equalsIgnoreCase(att.getStatus()) ||
//                            "onduty".equalsIgnoreCase(att.getStatus()))
//                    .count();
//
//            if (presentDays >= workingDays) {
//                perfectAttendanceEmployees.add(employee.getId());
//            }
//        }
//
//        return perfectAttendanceEmployees;
//    }
//
//    /**
//     * Get dashboard summary with today's statistics
//     */
//    public Map<String, Object> getDashboardSummary() {
//        LocalDate today = LocalDate.now();
//        List<Attendance> todayAttendance = attendanceConditionRepository.findByDate(today);
//
//        Map<String, Object> summary = new HashMap<>();
//        Map<String, Long> statusCounts = todayAttendance.stream()
//                .collect(Collectors.groupingBy(
//                        att -> att.getStatus() != null ? att.getStatus().toLowerCase() : "absent",
//                        Collectors.counting()
//                ));
//
//        summary.put("totalEmployees", employeeRepository.count());
//        summary.put("presentToday", statusCounts.getOrDefault("present", 0L));
//        summary.put("absentToday", statusCounts.getOrDefault("absent", 0L));
//        summary.put("lateToday", statusCounts.getOrDefault("late", 0L));
//        summary.put("onDutyToday", statusCounts.getOrDefault("onduty", 0L));
//        summary.put("halfDayToday", statusCounts.getOrDefault("halfday", 0L));
//        summary.put("permissionToday", statusCounts.getOrDefault("permission", 0L));
//        summary.put("date", today.toString());
//
//        return summary;
//    }
//
//    /**
//     * Get department-wise attendance statistics
//     */
//    public Map<String, Integer> getDepartmentAttendanceStats(String department, LocalDate startDate, LocalDate endDate) {
//        List<Attendance> departmentAttendance;
//
//        if (startDate != null && endDate != null) {
//            departmentAttendance = attendanceConditionRepository.findByDepartmentAndDateBetween(department, startDate, endDate);
//        } else {
//            departmentAttendance = attendanceConditionRepository.findByDepartment(department);
//        }
//
//        return calculateStatusSummary(departmentAttendance);
//    }
//
//    /**
//     * Export attendance data to CSV format
//     */
//    public String exportAttendanceToCSV(String employeeId, LocalDate startDate, LocalDate endDate) {
//        List<Attendance> attendanceRecords = attendanceConditionRepository
//                .findByEmployeeIdAndDateBetween(employeeId, startDate, endDate);
//
//        StringBuilder csv = new StringBuilder();
//        csv.append("Date,Day,Status,Check In,Check Out,Hours,Reason\n");
//
//        DateTimeFormatter dateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
//        DateTimeFormatter dayFormatter = DateTimeFormatter.ofPattern("EEE");
//
//        for (Attendance record : attendanceRecords) {
//            csv.append(record.getDate().format(dateFormatter)).append(",");
//            csv.append(record.getDate().format(dayFormatter)).append(",");
//            csv.append(record.getStatus() != null ? record.getStatus() : "absent").append(",");
//            csv.append(record.getCheckinTime() != null ? record.getCheckinTime().toString() : "-").append(",");
//            csv.append(record.getCheckoutTime() != null ? record.getCheckoutTime().toString() : "-").append(",");
//            csv.append(calculateWorkingHours(record.getCheckinTime(), record.getCheckoutTime())).append(",");
//            csv.append(record.getReason() != null ? record.getReason().replace(",", ";") : "-").append("\n");
//        }
//
//        return csv.toString();
//    }
//
//    /**
//     * Get attendance patterns analysis
//     */
//    public Map<String, Object> getAttendancePatterns(String employeeId, LocalDate startDate, LocalDate endDate) {
//        List<Attendance> records = attendanceConditionRepository.findByEmployeeIdAndDateBetween(employeeId, startDate, endDate);
//
//        Map<String, Object> patterns = new HashMap<>();
//
//        // Calculate average check-in time
//        OptionalDouble avgCheckinHour = records.stream()
//                .filter(r -> r.getCheckinTime() != null)
//                .mapToInt(r -> r.getCheckinTime().getHour())
//                .average();
//
//        patterns.put("averageCheckinHour", avgCheckinHour.orElse(0.0));
//
//        // Calculate most frequent status
//        Map<String, Long> statusFrequency = records.stream()
//                .collect(Collectors.groupingBy(
//                        r -> r.getStatus() != null ? r.getStatus() : "absent",
//                        Collectors.counting()
//                ));
//
//        String mostFrequentStatus = statusFrequency.entrySet().stream()
//                .max(Map.Entry.comparingByValue())
//                .map(Map.Entry::getKey)
//                .orElse("absent");
//
//        patterns.put("mostFrequentStatus", mostFrequentStatus);
//        patterns.put("statusBreakdown", statusFrequency);
//
//        return patterns;
//    }
//
//    // Private helper methods
//
//    private List<Attendance> getTodayAttendanceAsList(String employeeId) {
//        Attendance todayAttendance = getTodayAttendance(employeeId);
//        if (todayAttendance != null) {
//            return Arrays.asList(todayAttendance);
//        }
//        // Return empty list if no attendance record for today
//        return new ArrayList<>();
//    }
//
//    private List<Attendance> getWeeklyAttendance(String employeeId) {
//        LocalDate today = LocalDate.now();
//        LocalDate weekStart = today.with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY));
//        LocalDate weekEnd = today.with(TemporalAdjusters.nextOrSame(DayOfWeek.SUNDAY));
//
//        return attendanceConditionRepository.findByEmployeeIdAndDateBetween(employeeId, weekStart, weekEnd);
//    }
//
//    private List<Attendance> getMonthlyAttendance(String employeeId) {
//        LocalDate today = LocalDate.now();
//        LocalDate monthStart = today.with(TemporalAdjusters.firstDayOfMonth());
//        LocalDate monthEnd = today.with(TemporalAdjusters.lastDayOfMonth());
//
//        return attendanceConditionRepository.findByEmployeeIdAndDateBetween(employeeId, monthStart, monthEnd);
//    }
//
//    private Map<String, Integer> calculateStatusSummary(List<Attendance> attendanceList) {
//        Map<String, Integer> summary = new HashMap<>();
//
//        summary.put("present", 0);
//        summary.put("absent", 0);
//        summary.put("late", 0);
//        summary.put("onduty", 0);
//        summary.put("halfday", 0);
//        summary.put("permission", 0);
//        summary.put("total", attendanceList.size());
//
//        for (Attendance attendance : attendanceList) {
//            String status = attendance.getStatus() != null ? attendance.getStatus().toLowerCase() : "absent";
//            summary.put(status, summary.getOrDefault(status, 0) + 1);
//        }
//
//        return summary;
//    }
//
//    private String calculateWorkingHours(LocalTime checkIn, LocalTime checkOut) {
//        if (checkIn == null || checkOut == null) {
//            return "-";
//        }
//
//        try {
//            int hours = checkOut.getHour() - checkIn.getHour();
//            int minutes = checkOut.getMinute() - checkIn.getMinute();
//
//            if (minutes < 0) {
//                hours--;
//                minutes += 60;
//            }
//
//            if (hours < 0) {
//                hours += 24; // Handle next day checkout
//            }
//
//            return String.format("%dh %dm", hours, minutes);
//        } catch (Exception e) {
//            return "-";
//        }
//    }
//}