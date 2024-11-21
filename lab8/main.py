from dsmltf import build_tree_id3, classify

def main():
    inputs_1 = [
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Mid', 'lang': 'C++', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'C++', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'Go', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'Go', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Go', 'tweets': 'yes', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Senior', 'lang': 'C++', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Junior', 'lang': 'C++', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'JavaScript', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'JavaScript', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'JavaScript', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Senior', 'lang': 'Ruby', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Ruby', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'Ruby', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Senior', 'lang': 'Go', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Mid', 'lang': 'Go', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Go', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Senior', 'lang': 'PHP', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Mid', 'lang': 'PHP', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'PHP', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Junior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'C++', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'C++', 'tweets': 'yes', 'phd': 'no'}, True),
    ]
    
    tree = build_tree_id3(inputs_1)
    
    print(classify(tree,{'level':'Junior', 'lang':'Python', 'tweets':'no','phd':'no'}))
   
    students_data = [
        ({"program_number": "090304", "debts": "no", "attendance": "more 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "no", "active_in_events": "yes", "teacher_feedback": "excellent"}, True),
        ({"program_number": "090304", "debts": "yes",  "attendance": "less 30%", "vk_registered": "no", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "bad"}, False),
        ({"program_number": "090304", "debts": "no", "attendance": "50% to 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "yes", "active_in_events": "yes", "teacher_feedback": "good"}, True),
        ({"program_number": "100503", "debts": "yes", "attendance": "30% to 50%", "vk_registered": "yes", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "good"}, False),
        ({"program_number": "090304", "debts": "no",  "attendance": "less 30%", "vk_registered": "no", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "bad"}, False),
        ({"program_number": "100503", "debts": "no", "attendance": "more 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "yes", "active_in_events": "yes", "teacher_feedback": "excellent"}, True),
        ({"program_number": "090304", "debts": "yes",  "attendance": "30% to 50%", "vk_registered": "no", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "bad"}, False),
        ({"program_number": "090304", "debts": "no", "attendance": "50% to 80%", "vk_registered": "yes", "full_time_education": "no", "sport_active": "yes", "active_in_events": "yes", "teacher_feedback": "good"}, True),
        ({"program_number": "100503", "debts": "no",  "attendance": "more 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "good"}, True),
        ({"program_number": "090304", "debts": "no",  "attendance": "30% to 50%", "vk_registered": "yes", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "bad"}, False),
        ({"program_number": "090304", "debts": "yes",  "attendance": "less 30%", "vk_registered": "no", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "bad"}, False),
        ({"program_number": "100503", "debts": "no", "attendance": "50% to 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "no", "active_in_events": "yes", "teacher_feedback": "excellent"}, True),
        ({"program_number": "090304", "debts": "yes", "attendance": "30% to 50%", "vk_registered": "yes", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "good"}, False),
        ({"program_number": "090304", "debts": "no", "attendance": "more 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "no", "active_in_events": "yes", "teacher_feedback": "excellent"}, True),
        ({"program_number": "100503", "debts": "yes", "attendance": "30% to 50%", "vk_registered": "no", "full_time_education": "yes", "sport_active": "yes", "active_in_events": "no", "teacher_feedback": "good"}, True),
        ({"program_number": "090304", "debts": "no",  "attendance": "more 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "no", "active_in_events": "yes", "teacher_feedback": "good"}, True),
        ({"program_number": "090304", "debts": "yes", "attendance": "less 30%", "vk_registered": "no", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "bad"}, False),
        ({"program_number": "100503", "debts": "no", "attendance": "50% to 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "yes", "active_in_events": "yes", "teacher_feedback": "excellent"}, True),
        ({"program_number": "090304", "debts": "yes",  "attendance": "30% to 50%", "vk_registered": "no", "full_time_education": "no", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "bad"}, False),
        ({"program_number": "090304", "debts": "no", "attendance": "more 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "no", "active_in_events": "yes", "teacher_feedback": "excellent"}, True)
    ]
    tree_2 = build_tree_id3(students_data)
    print(classify(tree_2,{"program_number": "100503", "debts": "no", "attendance": "50% to 80%", "vk_registered": "yes", "full_time_education": "yes", "sport_active": "no", "active_in_events": "no", "teacher_feedback": "good"}))
    
if __name__ == "__main__":
    main()