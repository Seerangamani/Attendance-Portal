package com.Attendance.BackEnd_Attendance.Service;

import com.Attendance.BackEnd_Attendance.Model.Attendance;
import com.Attendance.BackEnd_Attendance.Model.Employee;
import com.Attendance.BackEnd_Attendance.Repository.AttendanceRepository;
import com.Attendance.BackEnd_Attendance.Repository.EmployeeRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;
import java.util.Base64;

@Service
public class AttendanceViewService {

    private final AttendanceRepository attendanceRepo;
    private final EmployeeRepository employeeRepo;

    public AttendanceViewService(AttendanceRepository attendanceRepo, EmployeeRepository employeeRepo) {
        this.attendanceRepo = attendanceRepo;
        this.employeeRepo = employeeRepo;
    }

    // 1. Get each employee with today's attendance status
    public List<Map<String, Object>> getEmployeeAttendanceForDate(LocalDate date) {
        List<Employee> employees = employeeRepo.findAll();
        List<Attendance> attendanceList = attendanceRepo.findAll();

        Map<String, Attendance> todayMap = attendanceList.stream()
                .filter(a -> date.equals(a.getDate()))
                .collect(Collectors.toMap(
                        Attendance::getEmployeeId,
                        a -> a,
                        (a1, a2) -> a1
                ));

        List<Map<String, Object>> result = new ArrayList<>();

        for (Employee emp : employees) {
            Map<String, Object> map = new HashMap<>();
            map.put("id", emp.getId());
            map.put("name", emp.getUsername());
            map.put("department", emp.getDepartment());
            map.put("designation", emp.getDesignation());

            // Convert BLOB image to Base64
            String base64Image = emp.getProfileImage() != null
                    ? Base64.getEncoder().encodeToString(emp.getProfileImage())
                    : "";
            map.put("photo", base64Image);

            Attendance today = todayMap.get(emp.getId());
            if (today != null) {
                map.put("status", today.getStatus());
                map.put("checkinTime", today.getCheckinTime());
                map.put("checkoutTime", today.getCheckoutTime());
                map.put("date", today.getDate().toString());
                map.put("reason", today.getReason());
            } else {
                map.put("status", "No Record");
                map.put("checkinTime", null);
                map.put("checkoutTime", null);
                map.put("date", date.toString());
                map.put("reason", null);
            }

            result.add(map);
        }
        return result;
    }

    // 2. Get today's status for a specific employee
    public Map<String, Object> getTodayStatusForEmployee(String employeeId) {
        LocalDate today = LocalDate.now();

        Optional<Employee> employeeOpt = employeeRepo.findById(employeeId);
        if (!employeeOpt.isPresent()) {
            throw new RuntimeException("Employee not found with ID: " + employeeId);
        }

        Employee employee = employeeOpt.get();

        // Find today's attendance record
        Attendance todayAttendance = attendanceRepo.findAll().stream()
                .filter(a -> a.getEmployeeId().equals(employeeId) && today.equals(a.getDate()))
                .findFirst()
                .orElse(null);

        Map<String, Object> result = new HashMap<>();
        result.put("id", employee.getId());
        result.put("name", employee.getUsername());
        result.put("date", today.toString());

        if (todayAttendance != null) {
            result.put("status", todayAttendance.getStatus());
            result.put("checkinTime", todayAttendance.getCheckinTime());
            result.put("checkoutTime", todayAttendance.getCheckoutTime());
            result.put("reason", todayAttendance.getReason());
        } else {
            result.put("status", "No Record");
            result.put("checkinTime", null);
            result.put("checkoutTime", null);
            result.put("reason", null);
        }

        return result;
    }

    // 3. Full attendance for employee (all records, sorted by date descending)
    public List<Attendance> getFullAttendanceForEmployee(String id) {
        return attendanceRepo.findAll().stream()
                .filter(a -> a.getEmployeeId().equals(id))
                .sorted((a1, a2) -> a2.getDate().compareTo(a1.getDate())) // Sort by date descending
                .collect(Collectors.toList());
    }

    // 4. Filter attendance by date range or overall
    public List<Attendance> filterAttendance(String id, String from, String to) {
        LocalDate fromDate;
        LocalDate toDate;

        // Parse dates if provided
        if (from != null && !from.trim().isEmpty()) {
            fromDate = LocalDate.parse(from);
        } else {
            fromDate = null;
        }
        if (to != null && !to.trim().isEmpty()) {
            toDate = LocalDate.parse(to);
        } else {
            toDate = null;
        }

        List<Attendance> allAttendance = attendanceRepo.findAll().stream()
                .filter(a -> a.getEmployeeId().equals(id))
                .collect(Collectors.toList());

        // Apply date filtering if dates are provided
        if (fromDate != null || toDate != null) {
            return allAttendance.stream()
                    .filter(a -> {
                        LocalDate attendanceDate = a.getDate();
                        boolean afterFrom = fromDate == null || !attendanceDate.isBefore(fromDate);
                        boolean beforeTo = toDate == null || !attendanceDate.isAfter(toDate);
                        return afterFrom && beforeTo;
                    })
                    .sorted((a1, a2) -> a2.getDate().compareTo(a1.getDate()))
                    .collect(Collectors.toList());
        }

        // Return all records sorted by date descending for overall attendance
        return allAttendance.stream()
                .sorted((a1, a2) -> a2.getDate().compareTo(a1.getDate()))
                .collect(Collectors.toList());
    }

    // 5. Get available months for an employee (for dropdown)
    public List<Map<String, Object>> getAvailableMonthsForEmployee(String employeeId) {
        List<Attendance> attendance = getFullAttendanceForEmployee(employeeId);

        Set<YearMonth> months = attendance.stream()
                .map(a -> YearMonth.from(a.getDate()))
                .collect(Collectors.toSet());

        return months.stream()
                .sorted(Comparator.reverseOrder()) // Latest months first
                .map(month -> {
                    Map<String, Object> monthData = new HashMap<>();
                    monthData.put("value", month.toString()); // YYYY-MM format
                    monthData.put("label", month.format(DateTimeFormatter.ofPattern("MMMM yyyy")));
                    monthData.put("year", month.getYear());
                    monthData.put("month", month.getMonthValue());
                    return monthData;
                })
                .collect(Collectors.toList());
    }

    // 6. Get attendance statistics for dashboard
    public Map<String, Object> getAttendanceStatistics(String employeeId) {
        List<Attendance> allAttendance = getFullAttendanceForEmployee(employeeId);

        Map<String, Long> statusCounts = allAttendance.stream()
                .filter(a -> a.getStatus() != null && !a.getStatus().trim().isEmpty())
                .collect(Collectors.groupingBy(
                        a -> a.getStatus().toLowerCase().trim(),
                        Collectors.counting()
                ));

        long totalRecords = allAttendance.size();
        long presentCount = statusCounts.getOrDefault("present", 0L) + statusCounts.getOrDefault("late", 0L);
        double overallPercentage = totalRecords > 0 ? Math.round((presentCount * 100.0) / totalRecords * 100.0) / 100.0 : 0.0;

        // Get current month statistics
        YearMonth currentMonth = YearMonth.now();
        List<Attendance> currentMonthAttendance = allAttendance.stream()
                .filter(a -> YearMonth.from(a.getDate()).equals(currentMonth))
                .collect(Collectors.toList());

        Map<String, Long> currentMonthCounts = currentMonthAttendance.stream()
                .filter(a -> a.getStatus() != null && !a.getStatus().trim().isEmpty())
                .collect(Collectors.groupingBy(
                        a -> a.getStatus().toLowerCase().trim(),
                        Collectors.counting()
                ));

        long currentMonthTotal = currentMonthAttendance.size();
        long currentMonthPresent = currentMonthCounts.getOrDefault("present", 0L) + currentMonthCounts.getOrDefault("late", 0L);
        double currentMonthPercentage = currentMonthTotal > 0 ? Math.round((currentMonthPresent * 100.0) / currentMonthTotal * 100.0) / 100.0 : 0.0;

        Map<String, Object> statistics = new HashMap<>();
        statistics.put("totalRecords", totalRecords);
        statistics.put("overallPercentage", overallPercentage);
        statistics.put("statusCounts", statusCounts);
        statistics.put("currentMonthTotal", currentMonthTotal);
        statistics.put("currentMonthPercentage", currentMonthPercentage);
        statistics.put("currentMonthCounts", currentMonthCounts);

        return statistics;
    }

    // 7. Calculate working hours between check-in and check-out
    public String calculateWorkingHours(String checkinTime, String checkoutTime) {
        if (checkinTime == null || checkoutTime == null ||
                checkinTime.trim().isEmpty() || checkoutTime.trim().isEmpty()) {
            return "00:00";
        }

        try {
            String[] checkinParts = checkinTime.split(":");
            String[] checkoutParts = checkoutTime.split(":");

            int checkinHour = Integer.parseInt(checkinParts[0]);
            int checkinMinute = Integer.parseInt(checkinParts[1]);
            int checkoutHour = Integer.parseInt(checkoutParts[0]);
            int checkoutMinute = Integer.parseInt(checkoutParts[1]);

            int totalCheckinMinutes = checkinHour * 60 + checkinMinute;
            int totalCheckoutMinutes = checkoutHour * 60 + checkoutMinute;

            int workingMinutes = totalCheckoutMinutes - totalCheckinMinutes;

            if (workingMinutes < 0) {
                workingMinutes += 24 * 60; // Handle overnight shifts
            }

            int hours = workingMinutes / 60;
            int minutes = workingMinutes % 60;

            return String.format("%02d:%02d", hours, minutes);
        } catch (Exception e) {
            return "00:00";
        }
    }
}