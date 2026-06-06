import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'learnx_user_behavior_dataset_10M.csv'
df = pd.read_csv(file_path)
#Bước 1
print(f"Số lượng dòng dữ liệu: {len(df)}")
print(df.info()) #Thuộc tính chính
print(df.describe()) #Thống kê mô tả 
#Bước 2
count_before = len(df)
print("\nGiá trị thiếu mỗi cột:")
print(df.isnull().sum())
df = df.dropna() 
print(f"Số lượng dòng dữ liệu sau khi làm sạch: {len(df)}")
print(f"Giá trị trùng lặp: {df.duplicated().sum()}")
df = df.drop_duplicates()
df = df[(df['age'] > 0) & (df['avg_session_minutes'] >= 0)]
count_after = len(df)
print(f"Số lượng dòng đã loại bỏ: {count_before - count_after}")
print(f"Số lượng bản ghi sau khi làm sạch: {count_after}")
#Bước 3
plt.figure(figsize=(20, 14)) 

plt.subplot(2, 2, 1)
sns.histplot(df['avg_session_minutes'], bins=50, kde=True, color='blue')
plt.title('Phân phối thời gian học trung bình (phút)', fontsize=14, pad=15)
plt.xlabel('Số phút/phiên', fontsize=12)

plt.subplot(2, 2, 2)
sns.countplot(x='sessions_per_week', data=df, hue='sessions_per_week', palette='viridis', legend=False)
plt.title('Tần suất truy cập mỗi tuần', fontsize=14, pad=15)
plt.xlabel('Số lần truy cập', fontsize=12)

plt.subplot(2, 2, 3)
sns.histplot(df['completion_rate'], bins=20, kde=True, color='green')
plt.title('Phân bổ tỷ lệ hoàn thành khóa học (%)', fontsize=14, pad=15)
plt.xlabel('Tỷ lệ hoàn thành (0.0 - 1.0)', fontsize=12)

plt.subplot(2, 2, 4)
df['month_group'] = df['signup_days_ago'] // 30
trend_data = df['month_group'].value_counts().sort_index(ascending=False)
trend_data.plot(kind='line', marker='o', color='orange', linewidth=2)
plt.title('Xu hướng người dùng mới theo tháng (Gần đây -> Cũ)', fontsize=14, pad=15)
plt.xlabel('Số tháng trước đây', fontsize=12)
plt.ylabel('Số lượng người dùng', fontsize=12)


plt.tight_layout(pad=5.0) 
plt.subplots_adjust(hspace=0.4, wspace=0.3) 

plt.show()
#Bước 4
top_watcher_threshold = df['videos_watched'].quantile(0.99)
power_users = df[df['videos_watched'] > top_watcher_threshold]
print(f"- Số lượng người dùng hoạt động cao: {len(power_users)}")

ghost_users = df[(df['courses_enrolled'] > 5) & (df['completion_rate'] < 10)]
print(f"- Số lượng đăng ký nhưng không hoàn thành: {len(ghost_users)}")

high_spenders = df[df['total_spent_usd'] > df['total_spent_usd'].quantile(0.99)]
print(f"- Số lượng chi tiêu cao bất thường: {len(high_spenders)}")


