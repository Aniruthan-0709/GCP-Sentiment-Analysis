import smtplib
from email.mime.text import MIMEText
import os

def send_email():
    sender_email = os.getenv("SMTP_SENDER")
    receiver_email = os.getenv("SMTP_RECEIVER")
    smtp_password = os.getenv("SMTP_PASSWORD")  # Gmail app password

    subject = "✅ Model Development Pipeline Completed"
    body = (
        "Hello,\n\n"
        "Your MLOps Model Dev Pipeline completed successfully!\n"
        "The Model is trainied and the model.pkl file was uploaded to the GCP bucket as expected.\n\n"
        "Best,\nMLOps Bot 🤖"
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, smtp_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

if __name__ == "__main__":
    send_email()
