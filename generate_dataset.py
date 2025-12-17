"""
Employee Attrition Dataset Generator v2.0
==========================================
Generates 500 synthetic employees with realistic attrition risk patterns.

Data Quality: Minimum 3 years historical records for meaningful analysis.

Distribution:
- 90% Normal employees (varied performance tiers)
- 10% At-Risk employees (3 archetypes: Burnout, Quiet Quitter, Stalled Career)

Enhanced with sophisticated risk factors:
- Career progression metrics
- Compensation benchmarking
- Manager engagement tracking
- Office presence patterns
- Health indicators (sick leave)
- Training & development
- Recognition & rewards
- Team dynamics
"""

import pandas as pd
import numpy as np
import json
import random
from typing import Tuple, List, Dict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

TOTAL_EMPLOYEES = 500
AT_RISK_PERCENTAGE = 0.10  # 10%

# At-risk archetype distribution (must sum to AT_RISK_PERCENTAGE)
ARCHETYPE_DISTRIBUTION = {
    'Burnout': 0.037,        # 3.7%
    'Quiet_Quitter': 0.032,  # 3.2%
    'Stalled_Career': 0.031  # 3.1%
}

# Normal employee tier distribution (must sum to 1.0)
NORMAL_TIER_DISTRIBUTION = {
    'Top_Performer': 0.15,    # 15%
    'Good_Performer': 0.35,   # 35%
    'Average': 0.35,          # 35%
    'Below_Average': 0.15     # 15%
}

DEPARTMENTS = ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing', 'Operations']
LEVELS = ['CEO', 'C-Level', 'Senior Manager', 'Manager', 'Standard', 'Junior']
LEVEL_WEIGHTS = [1, 4, 15, 40, 280, 160]  # Realistic org structure

# Market salary benchmarks by level (in thousands)
MARKET_SALARY_BY_LEVEL = {
    'CEO': 500,
    'C-Level': 300,
    'Senior Manager': 180,
    'Manager': 120,
    'Standard': 75,
    'Junior': 50
}

# =============================================================================
# NAME GENERATOR
# =============================================================================

FIRST_NAMES = [
    'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
    'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
    'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Lisa', 'Daniel', 'Nancy',
    'Matthew', 'Betty', 'Anthony', 'Margaret', 'Mark', 'Sandra', 'Donald', 'Ashley',
    'Steven', 'Kimberly', 'Paul', 'Emily', 'Andrew', 'Donna', 'Joshua', 'Michelle',
    'Kenneth', 'Dorothy', 'Kevin', 'Carol', 'Brian', 'Amanda', 'George', 'Melissa',
    'Timothy', 'Deborah', 'Ronald', 'Stephanie', 'Edward', 'Rebecca', 'Jason', 'Sharon',
    'Jeffrey', 'Laura', 'Ryan', 'Cynthia', 'Jacob', 'Kathleen', 'Gary', 'Amy',
    'Nicholas', 'Angela', 'Eric', 'Shirley', 'Jonathan', 'Anna', 'Stephen', 'Brenda',
    'Wei', 'Mei', 'Chen', 'Li', 'Wang', 'Zhang', 'Liu', 'Yang', 'Huang', 'Zhao',
    'Raj', 'Priya', 'Amit', 'Neha', 'Vikram', 'Ananya', 'Arjun', 'Divya',
    'Carlos', 'Maria', 'Juan', 'Ana', 'Pedro', 'Sofia', 'Miguel', 'Isabella'
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
    'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
    'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
    'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
    'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell',
    'Carter', 'Roberts', 'Chen', 'Wang', 'Li', 'Zhang', 'Liu', 'Yang', 'Patel',
    'Kumar', 'Singh', 'Sharma', 'Kim', 'Park', 'Choi', 'Tanaka', 'Suzuki', 'Yamamoto',
    'Mueller', 'Schmidt', 'Schneider', 'Fischer', 'Weber', 'Meyer', 'Wagner', 'Becker'
]

def generate_name() -> str:
    """Generate a random full name."""
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

# =============================================================================
# MANAGER ASSIGNMENT
# =============================================================================

# Create 2-3 managers per department
MANAGERS_BY_DEPARTMENT = {}
for dept in DEPARTMENTS:
    num_managers = random.randint(2, 3)
    managers = []
    for _ in range(num_managers):
        manager_name = generate_name()
        managers.append(manager_name)
    MANAGERS_BY_DEPARTMENT[dept] = managers

def get_manager_for_department(department: str) -> str:
    """Assign a random manager from the department's manager pool."""
    return random.choice(MANAGERS_BY_DEPARTMENT[department])

# =============================================================================
# TEXT TEMPLATES - Manager Notes & Survey Comments
# =============================================================================

MANAGER_NOTES_TEMPLATES = {
    'Top_Performer': [
        "Outstanding results this quarter. {name} consistently exceeds targets and demonstrates strong leadership potential. Team members look up to them for guidance and mentorship.",
        "Exceptional performance across all metrics. {name} has taken on additional responsibilities proactively and delivered results above expectations. A true asset to the team.",
        "{name} continues to be one of our strongest contributors. Their work quality is exemplary, and they've helped onboard three new team members successfully this quarter.",
        "Truly impressive quarter for {name}. They led the cross-functional initiative and delivered ahead of schedule. Their technical expertise and work ethic are remarkable.",
        "{name} has shown exceptional growth this year. Their ability to solve complex problems while maintaining excellent stakeholder relationships makes them invaluable.",
    ],
    'Good_Performer': [
        "{name} is a reliable team member who consistently meets expectations. Good collaboration with peers and positive attitude in team meetings. Solid contributor overall.",
        "Dependable work from {name} this quarter. They complete assignments on time and communicate effectively. Would benefit from taking on more stretch projects.",
        "{name} delivers solid results and works well with the team. Their attention to detail has improved, and they handle their workload efficiently. Good team player.",
        "Consistent performer. {name} meets deadlines reliably and maintains good working relationships. They contribute positively to team morale and culture.",
        "{name} has been a steady contributor this quarter. Their work quality is good, and they're responsive to feedback. Ready for more challenging assignments.",
    ],
    'Average': [
        "{name} is meeting basic expectations but has room for growth. Work is generally acceptable, though could benefit from taking more initiative on projects.",
        "Adequate performance from {name}. They complete required tasks but rarely go beyond the minimum. Would benefit from more proactive engagement in team activities.",
        "{name} is developing their skills gradually. Performance is satisfactory, though there's opportunity to improve time management and prioritization.",
        "Meeting expectations overall. {name} handles routine tasks well but sometimes struggles with more complex assignments. Coaching sessions have been scheduled.",
        "{name} shows potential but hasn't fully applied themselves yet. Work quality is acceptable. Encouraging them to participate more actively in team discussions.",
    ],
    'Below_Average': [
        "{name} is struggling to meet performance targets. We've discussed specific areas for improvement and created a development plan. Additional coaching required.",
        "Performance concerns noted for {name}. Missing deadlines has been an issue, and work quality needs improvement. Currently on a performance improvement plan.",
        "{name} needs significant support to meet role requirements. Their engagement has been inconsistent, and we're working together on specific skill development areas.",
        "Below expectations this quarter. {name} has struggled with workload management and deliverable quality. Regular check-ins scheduled to provide additional guidance.",
        "Performance improvement needed. {name} has potential but is not meeting current expectations. We've identified training opportunities and paired them with a mentor.",
    ],
    'Burnout': [
        "Results are excellent, but {name} looks visibly exhausted in recent weeks. They've been first in and last out every day. Concerned about sustainability of this pace.",
        "{name} continues to deliver outstanding work but showing signs of burnout. Dark circles, skipped lunches, working weekends consistently. Need to discuss workload.",
        "Top performer but worrying signs. {name} has seemed increasingly stressed and mentioned feeling overwhelmed. Despite great output, their energy seems depleted.",
        "{name}'s work quality remains exceptional but their physical appearance has changed noticeably. They seem tired constantly and have cancelled all vacation plans.",
        "Excellent deliverables from {name} but at what cost? They've been working extreme hours for months. Recent sick days suggest their body is showing the strain.",
    ],
    'Quiet_Quitter': [
        "{name} has become noticeably disengaged. Doing the bare minimum to get by, rarely speaks up in meetings, and peers report they're hard to reach on Slack.",
        "Concerned about {name}'s recent behavior shift. They used to be proactive but now only do exactly what's assigned. Camera off in all meetings, minimal responses.",
        "{name} seems to have mentally checked out. Work is technically acceptable but lacks any initiative or enthusiasm. Declining optional team activities.",
        "Significant change in {name}'s engagement level. They complete tasks but show no interest in going beyond. Colleagues mention difficulty getting responses from them.",
        "{name} is withdrawing from the team. Attendance is spotty, communication is minimal, and they've stopped participating in discussions. Need to have a conversation.",
    ],
    'Stalled_Career': [
        "{name} has expressed frustration about career progression multiple times. They're a reliable performer who feels stuck. Mentioned looking at external opportunities.",
        "Solid work from {name} but growing concerns about retention. They've been in the same role for years and feel overlooked for promotion despite good performance.",
        "{name} is a dependable contributor but their engagement has dropped since being passed over for promotion. They've asked about internal mobility options.",
        "Retention risk for {name}. They've been transparent about disappointment with career growth here. Good performer who feels their potential isn't being recognized.",
        "{name} mentioned updating their resume during a casual conversation. They feel their career has stalled and are frustrated by lack of advancement opportunities.",
    ]
}

SURVEY_COMMENTS_TEMPLATES = {
    'Top_Performer': [
        "I enjoy challenging work and the opportunity to make an impact. The team is great, and I feel my contributions are valued. Looking forward to growing with the company.",
        "This has been a rewarding quarter. I appreciate the autonomy I have in my role and the interesting projects. Would love more visibility into leadership decisions.",
        "I feel engaged and motivated. The work is challenging in a good way. I'd appreciate more recognition programs for high performers in the organization.",
        "Great team, interesting problems to solve. I'm learning a lot and feel supported by my manager. Would like to see more career development opportunities.",
        "Proud of what we've accomplished as a team. The culture is positive, and I feel empowered to do my best work. Excited about upcoming projects.",
    ],
    'Good_Performer': [
        "Generally satisfied with my role and team. Work-life balance has been manageable. Would appreciate more clarity on promotion criteria and timelines.",
        "I like my colleagues and the work is interesting most of the time. Communication from leadership could be better. Overall a good place to work.",
        "Things are going well. I feel adequately supported and have good relationships with my team. Would like more opportunities for skill development.",
        "Decent work environment and reasonable expectations. I appreciate the flexibility. Some processes could be more efficient but nothing major.",
        "Happy with my team and manager. The work is steady and I'm learning new things. Would like more feedback on my performance and growth areas.",
    ],
    'Average': [
        "The job is okay. Some days are more engaging than others. I'm not unhappy but not particularly excited either. Could use more variety in my work.",
        "It's fine here. I do my job and go home. Would appreciate more interesting projects and better communication about company direction.",
        "Work is manageable but not very inspiring. I show up, do what's needed, and leave. Not sure about long-term growth opportunities here.",
        "Neutral feelings overall. Some good aspects, some frustrating ones. The bureaucracy slows things down. Would like to see more innovation encouraged.",
        "Adequate work environment. Not the best, not the worst. I wish there were more team-building activities and recognition for day-to-day contributions.",
    ],
    'Below_Average': [
        "Struggling to stay motivated. The work doesn't align well with my interests, and I'm not getting the support I need to improve. Feeling frustrated.",
        "Finding it hard to meet expectations. The training has been insufficient, and deadlines are often unrealistic. Need more guidance from management.",
        "Not my best period. I'm having trouble keeping up with the workload and expectations. Would benefit from clearer priorities and more structured onboarding.",
        "Feeling overwhelmed and unsupported. The goals seem arbitrary and the feedback inconsistent. I want to do better but need more help getting there.",
        "Difficult quarter. I know I need to improve but the path forward isn't clear. Would appreciate more constructive feedback and development resources.",
    ],
    'Burnout': [
        "I love the work but I'm exhausted. Haven't taken a real break in months. The pace is unsustainable but I feel like I can't slow down without consequences.",
        "Running on empty. The expectations keep increasing and there's no end in sight. I'm proud of my results but my health is suffering. Something has to give.",
        "Feeling burned out and stretched too thin. I've been working nights and weekends for months. The workload distribution on our team is not sustainable.",
        "I can't keep this up much longer. Every project is urgent, every deadline is critical. I haven't had a vacation in over a year. Need to find balance.",
        "Physically and mentally drained. The work is rewarding but the volume is crushing me. I've been sick more often lately and can't seem to recover fully.",
    ],
    'Quiet_Quitter': [
        "Just here for the paycheck at this point. I used to care more but realized it doesn't matter how much effort I put in. Doing what's required, nothing more.",
        "I've stopped going above and beyond. The company doesn't recognize extra effort anyway, so why bother? I do my job and clock out mentally and physically.",
        "Not invested anymore. The politics and lack of appreciation wore me down. I show up, do the minimum, and save my energy for life outside work.",
        "I've checked out emotionally. There's no point in trying harder when the system doesn't reward it. Collecting my salary until something better comes along.",
        "Lost interest in going the extra mile. Used to be engaged but realized it's not worth it. Now I just coast through and focus on my life outside work.",
    ],
    'Stalled_Career': [
        "Frustrated with the lack of growth here. I've been doing the same job for years while watching others get promoted. My contributions seem to go unnoticed.",
        "Feeling stuck and undervalued. I've asked about advancement multiple times but nothing changes. Starting to think my future isn't here.",
        "Career has completely stalled. No promotions, no new challenges, no recognition. I'm exploring external options because I don't see a path forward here.",
        "Disappointed with promotion decisions. I've delivered consistently for years but keep getting passed over. It's hard to stay motivated when there's no upward mobility.",
        "Time to move on probably. I've given this company years of solid work and have nothing to show for it. My peers at other companies have advanced while I'm stuck.",
    ]
}

# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def generate_daily_log(avg_hours: float, variance: float = 0.8) -> List[float]:
    """Generate 60 days of work hours with realistic patterns."""
    hours = []
    for day in range(60):
        day_of_week = day % 7
        if day_of_week in [5, 6]:  # Weekend
            base = avg_hours * 0.3 if random.random() < 0.2 else 0
        else:
            base = avg_hours
        
        daily_hours = max(0, np.random.normal(base, variance))
        daily_hours = min(daily_hours, 16)
        hours.append(round(daily_hours, 1))
    
    return hours

def generate_performance_history(base_score: int, trend: str = 'stable', years: int = 3) -> List[int]:
    """Generate quarterly performance scores with specified trend (3 years = 12 quarters)."""
    quarters = years * 4
    scores = []
    current = base_score
    
    for q in range(quarters):
        noise = random.randint(-3, 3)
        score = max(1, min(100, current + noise))
        scores.append(score)
        
        if trend == 'declining':
            current = max(40, current - random.randint(1, 3))
        elif trend == 'improving':
            current = min(95, current + random.randint(1, 2))
        elif trend == 'stable':
            current = current + random.randint(-1, 1)
    
    return scores

def generate_engagement_history(base_score: int, trend: str = 'stable', years: int = 3) -> List[int]:
    """Generate quarterly engagement scores (1-10) with specified trend (3 years = 12 quarters)."""
    quarters = years * 4
    scores = []
    current = base_score
    
    for q in range(quarters):
        noise = random.randint(-1, 1)
        score = max(1, min(10, current + noise))
        scores.append(score)
        
        if trend == 'declining':
            current = max(1, current - random.choice([0, 0, 1, 1, 2]))
        elif trend == 'improving':
            current = min(10, current + random.choice([0, 0, 1]))
        elif trend == 'stable':
            current = current + random.choice([-1, 0, 0, 0, 1])
    
    return scores

def generate_manager_1on1_history(base_freq: int, trend: str = 'stable', months: int = 12) -> List[int]:
    """Generate monthly 1:1 meeting counts (typical: 2-4 per month)."""
    counts = []
    current = base_freq
    
    for m in range(months):
        noise = random.randint(-1, 1)
        count = max(0, min(8, current + noise))
        counts.append(count)
        
        if trend == 'declining':
            current = max(0, current - random.choice([0, 0, 0, 1]))
        elif trend == 'improving':
            current = min(6, current + random.choice([0, 0, 1]))
    
    return counts

def generate_office_presence(base_pct: float, trend: str = 'stable', weeks: int = 12) -> List[float]:
    """Generate weekly office presence percentage (0-100%)."""
    presence = []
    current = base_pct
    
    for w in range(weeks):
        noise = random.uniform(-5, 5)
        pct = max(0, min(100, current + noise))
        presence.append(round(pct, 1))
        
        if trend == 'declining':
            current = max(10, current - random.uniform(0, 3))
        elif trend == 'improving':
            current = min(100, current + random.uniform(0, 2))
    
    return presence

def generate_sick_leave_history(base_days: float, trend: str = 'stable', months: int = 12) -> List[int]:
    """Generate monthly sick days taken."""
    days = []
    current = base_days
    
    for m in range(months):
        # Most months are 0, occasional sick days
        if random.random() < 0.3:  # 30% chance of sick day in a month
            count = max(0, int(np.random.poisson(current)))
        else:
            count = 0
        days.append(min(count, 5))  # Cap at 5 per month
        
        if trend == 'increasing':
            current = min(3, current + random.uniform(0, 0.2))
        elif trend == 'decreasing':
            current = max(0.5, current - random.uniform(0, 0.1))
    
    return days

def calculate_salary(level: str, compensation_gap_pct: float) -> Tuple[int, int]:
    """Calculate actual salary based on level and compensation gap."""
    market_salary = MARKET_SALARY_BY_LEVEL[level]
    actual_salary = int(market_salary * (1 + compensation_gap_pct / 100))
    return actual_salary, market_salary

def generate_normal_employee(employee_id: int, tier: str) -> Dict:
    """Generate a normal (not at-risk) employee with tier-specific attributes."""
    name = generate_name()
    level = random.choices(LEVELS, weights=LEVEL_WEIGHTS)[0]
    
    tier_config = {
        'Top_Performer': {
            'performance_base': random.randint(85, 95),
            'engagement_base': random.randint(8, 10),
            'avg_hours': round(random.uniform(8.0, 9.5), 1),
            'vacation': random.randint(12, 20),
            'sick_days_60d': random.randint(0, 2),
            'tenure': round(random.uniform(3, 10), 1),  # Min 3 years
            'months_since_promo': random.randint(6, 18),
            'comp_gap': random.uniform(0, 15),  # At or above market
            'manager_1on1_freq': random.randint(3, 4),
            'office_presence': random.uniform(70, 95),
            'training_months_ago': random.randint(1, 6),
            'recognition_months_ago': random.randint(1, 4),
            'team_departures_6m': random.randint(0, 1),
            'meeting_attendance_pct': random.uniform(85, 98),
            'response_time_hours': random.uniform(0.5, 2),
        },
        'Good_Performer': {
            'performance_base': random.randint(70, 84),
            'engagement_base': random.randint(6, 8),
            'avg_hours': round(random.uniform(7.5, 8.5), 1),
            'vacation': random.randint(10, 18),
            'sick_days_60d': random.randint(0, 3),
            'tenure': round(random.uniform(3, 8), 1),
            'months_since_promo': random.randint(8, 24),
            'comp_gap': random.uniform(-5, 10),
            'manager_1on1_freq': random.randint(2, 4),
            'office_presence': random.uniform(60, 85),
            'training_months_ago': random.randint(2, 12),
            'recognition_months_ago': random.randint(2, 8),
            'team_departures_6m': random.randint(0, 2),
            'meeting_attendance_pct': random.uniform(75, 92),
            'response_time_hours': random.uniform(1, 4),
        },
        'Average': {
            'performance_base': random.randint(55, 69),
            'engagement_base': random.randint(5, 7),
            'avg_hours': round(random.uniform(7.0, 8.0), 1),
            'vacation': random.randint(8, 15),
            'sick_days_60d': random.randint(1, 4),
            'tenure': round(random.uniform(3, 6), 1),
            'months_since_promo': random.randint(12, 30),
            'comp_gap': random.uniform(-10, 5),
            'manager_1on1_freq': random.randint(2, 3),
            'office_presence': random.uniform(50, 75),
            'training_months_ago': random.randint(6, 18),
            'recognition_months_ago': random.randint(4, 12),
            'team_departures_6m': random.randint(0, 2),
            'meeting_attendance_pct': random.uniform(65, 85),
            'response_time_hours': random.uniform(2, 6),
        },
        'Below_Average': {
            'performance_base': random.randint(40, 54),
            'engagement_base': random.randint(4, 6),
            'avg_hours': round(random.uniform(6.5, 7.5), 1),
            'vacation': random.randint(5, 12),
            'sick_days_60d': random.randint(2, 5),
            'tenure': round(random.uniform(3, 5), 1),
            'months_since_promo': random.randint(12, 36),
            'comp_gap': random.uniform(-15, 0),
            'manager_1on1_freq': random.randint(1, 3),
            'office_presence': random.uniform(40, 65),
            'training_months_ago': random.randint(12, 24),
            'recognition_months_ago': random.randint(8, 18),
            'team_departures_6m': random.randint(0, 3),
            'meeting_attendance_pct': random.uniform(55, 75),
            'response_time_hours': random.uniform(4, 12),
        }
    }
    
    config = tier_config[tier]
    actual_salary, market_salary = calculate_salary(level, config['comp_gap'])
    department = random.choice(DEPARTMENTS)
    manager_name = get_manager_for_department(department)
    
    return {
        'Employee_ID': f'EMP{employee_id:03d}',
        'Name': name,
        'Department': department,
        'Manager_Name': manager_name,
        'Level': level,
        'Tenure_Years': config['tenure'],
        'Months_Since_Promotion': config['months_since_promo'],
        
        # Time series data (JSON)
        'Daily_Log_JSON': json.dumps(generate_daily_log(config['avg_hours'])),
        'Performance_History_JSON': json.dumps(generate_performance_history(config['performance_base'], 'stable', 3)),
        'Engagement_History_JSON': json.dumps(generate_engagement_history(config['engagement_base'], 'stable', 3)),
        'Manager_1on1_History_JSON': json.dumps(generate_manager_1on1_history(config['manager_1on1_freq'], 'stable')),
        'Office_Presence_JSON': json.dumps(generate_office_presence(config['office_presence'], 'stable')),
        'Sick_Leave_History_JSON': json.dumps(generate_sick_leave_history(1.0, 'stable')),
        
        # Snapshot metrics
        'Sick_Days_60D': config['sick_days_60d'],
        'Vacation_Days_12M': config['vacation'],
        'Salary_Actual': actual_salary,
        'Salary_Market': market_salary,
        'Compensation_Gap_Pct': round(config['comp_gap'], 1),
        'Months_Since_Training': config['training_months_ago'],
        'Months_Since_Recognition': config['recognition_months_ago'],
        'Team_Departures_6M': config['team_departures_6m'],
        'Meeting_Attendance_Pct': round(config['meeting_attendance_pct'], 1),
        'Avg_Response_Time_Hours': round(config['response_time_hours'], 1),
        
        # Text data
        'Manager_Notes': random.choice(MANAGER_NOTES_TEMPLATES[tier]).format(name=name.split()[0]),
        'Survey_Comments': random.choice(SURVEY_COMMENTS_TEMPLATES[tier]),
        
        # Labels
        'Archetype': 'Normal',
        'At_Risk': 0
    }

def generate_burnout_employee(employee_id: int) -> Dict:
    """Generate a Burnout archetype employee (high performer, unsustainable pace)."""
    name = generate_name()
    level = random.choices(LEVELS[2:5], weights=[15, 40, 200])[0]
    
    # High performance with slight recent dip
    perf_history = generate_performance_history(random.randint(90, 98), 'stable', 3)
    perf_history[-2:] = [random.randint(85, 92), random.randint(82, 90)]  # Recent dip
    
    # Engagement declining recently
    eng_history = generate_engagement_history(8, 'stable', 3)
    eng_history[-4:] = [7, 6, 5, 4]  # Recent decline
    
    # Manager meetings declining (manager may be avoiding the conversation)
    manager_history = generate_manager_1on1_history(4, 'declining')
    
    # Office presence very high (always there)
    office_presence = generate_office_presence(95, 'stable')
    
    # Sick leave increasing
    sick_history = generate_sick_leave_history(0.5, 'increasing')
    sick_history[-3:] = [random.randint(1, 3), random.randint(2, 4), random.randint(2, 5)]
    
    actual_salary, market_salary = calculate_salary(level, random.uniform(5, 15))
    department = random.choice(DEPARTMENTS)
    manager_name = get_manager_for_department(department)
    
    return {
        'Employee_ID': f'EMP{employee_id:03d}',
        'Name': name,
        'Department': department,
        'Manager_Name': manager_name,
        'Level': level,
        'Tenure_Years': round(random.uniform(3, 7), 1),
        'Months_Since_Promotion': random.randint(12, 24),
        
        'Daily_Log_JSON': json.dumps(generate_daily_log(random.uniform(10, 12), variance=1.2)),
        'Performance_History_JSON': json.dumps(perf_history),
        'Engagement_History_JSON': json.dumps(eng_history),
        'Manager_1on1_History_JSON': json.dumps(manager_history),
        'Office_Presence_JSON': json.dumps(office_presence),
        'Sick_Leave_History_JSON': json.dumps(sick_history),
        
        'Sick_Days_60D': random.randint(5, 12),
        'Vacation_Days_12M': random.randint(0, 3),
        'Salary_Actual': actual_salary,
        'Salary_Market': market_salary,
        'Compensation_Gap_Pct': round((actual_salary - market_salary) / market_salary * 100, 1),
        'Months_Since_Training': random.randint(8, 18),
        'Months_Since_Recognition': random.randint(3, 8),
        'Team_Departures_6M': random.randint(1, 3),
        'Meeting_Attendance_Pct': round(random.uniform(92, 100), 1),
        'Avg_Response_Time_Hours': round(random.uniform(0.2, 1), 1),
        
        'Manager_Notes': random.choice(MANAGER_NOTES_TEMPLATES['Burnout']).format(name=name.split()[0]),
        'Survey_Comments': random.choice(SURVEY_COMMENTS_TEMPLATES['Burnout']),
        
        'Archetype': 'Burnout',
        'At_Risk': 1
    }

def generate_quiet_quitter_employee(employee_id: int) -> Dict:
    """Generate a Quiet Quitter archetype employee (disengaged, minimum effort)."""
    name = generate_name()
    level = random.choices(LEVELS[3:], weights=[30, 150, 80])[0]
    
    # Performance declining
    perf_history = generate_performance_history(random.randint(65, 75), 'declining', 3)
    
    # Engagement steep decline
    eng_history = [8, 8, 7, 7, 6, 6, 5, 5, 4, 3, 2, 2]
    
    # Manager meetings declining
    manager_history = generate_manager_1on1_history(3, 'declining')
    manager_history[-3:] = [1, 1, 0]
    
    # Office presence declining
    office_presence = generate_office_presence(70, 'declining')
    office_presence[-4:] = [random.uniform(30, 45) for _ in range(4)]
    
    # Sick leave moderate/increasing (mental health days)
    sick_history = generate_sick_leave_history(1.0, 'increasing')
    
    actual_salary, market_salary = calculate_salary(level, random.uniform(-15, -5))
    department = random.choice(DEPARTMENTS)
    manager_name = get_manager_for_department(department)
    
    return {
        'Employee_ID': f'EMP{employee_id:03d}',
        'Name': name,
        'Department': department,
        'Manager_Name': manager_name,
        'Level': level,
        'Tenure_Years': round(random.uniform(3, 5), 1),
        'Months_Since_Promotion': random.randint(18, 36),
        
        'Daily_Log_JSON': json.dumps(generate_daily_log(random.uniform(6.8, 7.3), variance=0.3)),
        'Performance_History_JSON': json.dumps(perf_history),
        'Engagement_History_JSON': json.dumps(eng_history),
        'Manager_1on1_History_JSON': json.dumps(manager_history),
        'Office_Presence_JSON': json.dumps(office_presence),
        'Sick_Leave_History_JSON': json.dumps(sick_history),
        
        'Sick_Days_60D': random.randint(3, 7),
        'Vacation_Days_12M': random.randint(8, 15),
        'Salary_Actual': actual_salary,
        'Salary_Market': market_salary,
        'Compensation_Gap_Pct': round((actual_salary - market_salary) / market_salary * 100, 1),
        'Months_Since_Training': random.randint(12, 24),
        'Months_Since_Recognition': random.randint(10, 18),
        'Team_Departures_6M': random.randint(2, 4),
        'Meeting_Attendance_Pct': round(random.uniform(45, 65), 1),
        'Avg_Response_Time_Hours': round(random.uniform(8, 24), 1),
        
        'Manager_Notes': random.choice(MANAGER_NOTES_TEMPLATES['Quiet_Quitter']).format(name=name.split()[0]),
        'Survey_Comments': random.choice(SURVEY_COMMENTS_TEMPLATES['Quiet_Quitter']),
        
        'Archetype': 'Quiet_Quitter',
        'At_Risk': 1
    }

def generate_stalled_career_employee(employee_id: int) -> Dict:
    """Generate a Stalled Career archetype employee (long tenure, no advancement)."""
    name = generate_name()
    level = random.choices(LEVELS[3:5], weights=[60, 200])[0]
    
    # Performance stable/good
    perf_history = generate_performance_history(random.randint(72, 82), 'stable', 3)
    
    # Engagement consistently low
    eng_history = [random.randint(3, 5) for _ in range(12)]
    
    # Manager meetings declining (frustrated employee)
    manager_history = generate_manager_1on1_history(3, 'declining')
    
    # Office presence declining
    office_presence = generate_office_presence(65, 'declining')
    
    # Sick leave normal
    sick_history = generate_sick_leave_history(1.0, 'stable')
    
    actual_salary, market_salary = calculate_salary(level, random.uniform(-18, -8))
    department = random.choice(DEPARTMENTS)
    manager_name = get_manager_for_department(department)
    
    return {
        'Employee_ID': f'EMP{employee_id:03d}',
        'Name': name,
        'Department': department,
        'Manager_Name': manager_name,
        'Level': level,
        'Tenure_Years': round(random.uniform(4, 10), 1),
        'Months_Since_Promotion': random.randint(36, 60),
        
        'Daily_Log_JSON': json.dumps(generate_daily_log(random.uniform(7.5, 8.2), variance=0.5)),
        'Performance_History_JSON': json.dumps(perf_history),
        'Engagement_History_JSON': json.dumps(eng_history),
        'Manager_1on1_History_JSON': json.dumps(manager_history),
        'Office_Presence_JSON': json.dumps(office_presence),
        'Sick_Leave_History_JSON': json.dumps(sick_history),
        
        'Sick_Days_60D': random.randint(1, 4),
        'Vacation_Days_12M': random.randint(10, 18),
        'Salary_Actual': actual_salary,
        'Salary_Market': market_salary,
        'Compensation_Gap_Pct': round((actual_salary - market_salary) / market_salary * 100, 1),
        'Months_Since_Training': random.randint(18, 36),
        'Months_Since_Recognition': random.randint(12, 24),
        'Team_Departures_6M': random.randint(1, 3),
        'Meeting_Attendance_Pct': round(random.uniform(60, 80), 1),
        'Avg_Response_Time_Hours': round(random.uniform(3, 8), 1),
        
        'Manager_Notes': random.choice(MANAGER_NOTES_TEMPLATES['Stalled_Career']).format(name=name.split()[0]),
        'Survey_Comments': random.choice(SURVEY_COMMENTS_TEMPLATES['Stalled_Career']),
        
        'Archetype': 'Stalled_Career',
        'At_Risk': 1
    }

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_dataset() -> pd.DataFrame:
    """Generate the complete employee attrition dataset."""
    employees = []
    employee_id = 1
    
    burnout_count = int(TOTAL_EMPLOYEES * ARCHETYPE_DISTRIBUTION['Burnout'])
    quiet_quitter_count = int(TOTAL_EMPLOYEES * ARCHETYPE_DISTRIBUTION['Quiet_Quitter'])
    stalled_career_count = int(TOTAL_EMPLOYEES * ARCHETYPE_DISTRIBUTION['Stalled_Career'])
    
    total_at_risk = burnout_count + quiet_quitter_count + stalled_career_count
    normal_count = TOTAL_EMPLOYEES - total_at_risk
    
    print(f"Generating {TOTAL_EMPLOYEES} employees:")
    print(f"  - Normal employees: {normal_count} (90%)")
    print(f"  - At-Risk (Burnout): {burnout_count} (3.7%)")
    print(f"  - At-Risk (Quiet Quitter): {quiet_quitter_count} (3.2%)")
    print(f"  - At-Risk (Stalled Career): {stalled_career_count} (3.1%)")
    print()
    
    print("Generating Burnout employees...")
    for _ in range(burnout_count):
        employees.append(generate_burnout_employee(employee_id))
        employee_id += 1
    
    print("Generating Quiet Quitter employees...")
    for _ in range(quiet_quitter_count):
        employees.append(generate_quiet_quitter_employee(employee_id))
        employee_id += 1
    
    print("Generating Stalled Career employees...")
    for _ in range(stalled_career_count):
        employees.append(generate_stalled_career_employee(employee_id))
        employee_id += 1
    
    print("Generating Normal employees...")
    tier_counts = {
        tier: int(normal_count * pct) 
        for tier, pct in NORMAL_TIER_DISTRIBUTION.items()
    }
    
    total_tier = sum(tier_counts.values())
    if total_tier < normal_count:
        tier_counts['Good_Performer'] += normal_count - total_tier
    
    for tier, count in tier_counts.items():
        print(f"  - {tier}: {count}")
        for _ in range(count):
            employees.append(generate_normal_employee(employee_id, tier))
            employee_id += 1
    
    random.shuffle(employees)
    df = pd.DataFrame(employees)
    df['Employee_ID'] = [f'EMP{i:03d}' for i in range(1, len(df) + 1)]
    
    return df

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the generated dataset for consistency."""
    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)
    
    print(f"\nTotal employees: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print(f"\nArchetype distribution:")
    print(df['Archetype'].value_counts())
    
    print(f"\nAt-Risk distribution:")
    print(df['At_Risk'].value_counts())
    
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nWarning - Missing values found:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values.")
    
    print("\nValidating JSON columns...")
    json_cols = [c for c in df.columns if 'JSON' in c]
    for col in json_cols:
        try:
            df[col].apply(json.loads)
            print(f"  {col}: Valid")
        except Exception as e:
            print(f"  {col}: ERROR - {e}")
    
    print(f"\nTenure range: {df['Tenure_Years'].min():.1f} - {df['Tenure_Years'].max():.1f} years")
    print(f"Compensation gap range: {df['Compensation_Gap_Pct'].min():.1f}% - {df['Compensation_Gap_Pct'].max():.1f}%")

if __name__ == "__main__":
    df = generate_dataset()
    validate_dataset(df)
    
    # Remove Archetype column from output - let the prediction tool discover patterns
    df_output = df.drop(columns=['Archetype'])
    
    output_path = "AI Class Project/employee_attrition_dataset_final.csv"
    df_output.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Dataset saved to: {output_path}")
    print(f"{'='*60}")
    
    print("\nColumn list:")
    for col in df_output.columns:
        print(f"  - {col}")
