"""Generate sample parking knowledge base PDFs for the InstaParkAI RAG system."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fpdf import FPDF


class ParkingPDF(FPDF):
    """Custom PDF with InstaParkAI branding."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "InstaParkAI - Smart Parking Platform", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, "System Manual & Specifications Guide", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def section_body(self, body: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, body)
        self.ln(4)


def create_system_overview_pdf(output_dir: Path) -> Path:
    """Create system_overview.pdf."""
    pdf = ParkingPDF()

    # Page 1 - Company Overview & Products
    pdf.add_page()
    pdf.section_title("Company Overview")
    pdf.section_body(
        "InstaParkAI is an industry-leading, AI-Powered Smart Parking Platform designed to "
        "revolutionize urban parking management. By combining state-of-the-art computer vision, "
        "smart sensors, and cloud-based automation, InstaParkAI maximizes slot utilization, "
        "reduces congestion, and enhances customer satisfaction. Our solutions are deployed across "
        "shopping malls, airports, corporate offices, and municipal parking lots."
    )

    pdf.section_title("Hardware Products")
    pdf.section_body(
        "1. ANPR Cameras: High-definition edge AI cameras featuring Automatic Number Plate Recognition "
        "with 99.8% detection accuracy under varying lighting conditions.\n"
        "2. RFID Readers: Long-range Radio Frequency Identification readers for seamless, barrier-free "
        "access control for registered users and VIPs.\n"
        "3. IoT Sensors: Low-power ultrasonic and geomagnetic occupancy sensors providing real-time "
        "parking slot occupancy status with a 10-year battery life."
    )

    # Page 2 - Software & Tech Stack
    pdf.add_page()
    pdf.section_title("Software Services")
    pdf.section_body(
        "InstaParkAI provides a comprehensive software suite consisting of:\n"
        "- Smart Parking Booking App: Allowing users to search, book, navigate to, and pay for parking slots in advance.\n"
        "- Operations Dashboard: Providing real-time analytics, revenue reporting, dynamic pricing rules, and hardware health metrics for parking operators.\n"
        "- Integration APIs: Standard REST APIs and SDKs enabling seamless connection with municipal systems and payment gateways."
    )

    pdf.section_title("Technology Stack")
    pdf.section_body(
        "Our system architecture is built on a modern, highly scalable technology stack:\n"
        "- Edge Computing: Local inference on smart cameras for instant plate recognition.\n"
        "- RAG Pipeline: BAAI/bge-m3 embeddings and FAISS + BM25 hybrid index for fast knowledge retrieval.\n"
        "- Generative AI: ChatGoogleGenerativeAI using Gemini 2.5 Flash model for natural language queries."
    )

    path = output_dir / "system_overview.pdf"
    pdf.output(str(path))
    return path


def create_pricing_and_amc_pdf(output_dir: Path) -> Path:
    """Create pricing_and_amc.pdf."""
    pdf = ParkingPDF()

    # Page 1 - Pricing Models
    pdf.add_page()
    pdf.section_title("Pricing Models")
    pdf.section_body(
        "InstaParkAI offers flexible pricing models designed to suit different operator sizes:\n"
        "1. SaaS Subscription: Monthly tier-based licensing starting from $199 per month for up to 50 slots, "
        "going up to $999 per month for large enterprise complexes. This includes access to the Operations Dashboard, "
        "regular software updates, and 24/7 customer support.\n"
        "2. Hardware Purchase: Upfront procurement of ANPR cameras, RFID readers, and IoT occupancy sensors. "
        "Volume-based discounts are available for installations exceeding 200 parking slots."
    )

    pdf.section_title("Contract Models")
    pdf.section_body(
        "We support both annual and multi-year contract options. Annual contracts offer standard pricing, "
        "while 3-year and 5-year contracts lock in subscription rates with a 15% discount and complimentary "
        "hardware replacement options."
    )

    # Page 2 - AMC Services
    pdf.add_page()
    pdf.section_title("AMC Services (Annual Maintenance Contract)")
    pdf.section_body(
        "To ensure uninterrupted system availability, InstaParkAI provides comprehensive AMC service packages:\n"
        "- Standard AMC: Includes quarterly preventive maintenance visits, remote technical support, and standard "
        "hardware repair with a 48-hour SLA.\n"
        "- Premium AMC: Includes monthly maintenance visits, 24/7 priority technical support, free replacement "
        "for defective hardware parts, and an expedited 4-hour SLA.\n"
        "- Critical Software Updates: All AMC contracts guarantee immediate installation of security patches "
        "and feature upgrades."
    )

    path = output_dir / "pricing_and_amc.pdf"
    pdf.output(str(path))
    return path


def create_faq_pdf(output_dir: Path) -> Path:
    """Create faq.pdf."""
    pdf = ParkingPDF()

    pdf.add_page()
    pdf.section_title("Frequently Asked Questions")

    pdf.section_body(
        "Q: What is ANPR technology?\n"
        "A: ANPR stands for Automatic Number Plate Recognition. It uses high-definition cameras and optical "
        "character recognition (OCR) to automatically read and record vehicle license plates at entry and exit points.\n\n"
        "Q: How do IoT sensors help in smart parking?\n"
        "A: Low-power IoT occupancy sensors are installed on each parking slot. They detect whether a slot is empty or "
        "occupied and relay this status to the booking app and operations dashboard in real time.\n\n"
        "Q: What pricing models are available?\n"
        "A: We offer a SaaS subscription tier model for software access, alongside upfront hardware purchase plans.\n\n"
        "Q: What are the benefits of smart parking operations?\n"
        "A: Smart parking reduces traffic congestion, minimizes vehicle emissions, increases revenue for operators by "
        "optimizing space utilization, and provides drivers with a hassle-free parking experience."
    )

    path = output_dir / "faq.pdf"
    pdf.output(str(path))
    return path


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "uploads"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating parking knowledge base PDFs...")
    files = [
        create_system_overview_pdf(output_dir),
        create_pricing_and_amc_pdf(output_dir),
        create_faq_pdf(output_dir),
    ]
    for f in files:
        print(f"  Created: {f}")
    print(f"\nAll {len(files)} PDFs created in {output_dir}")


if __name__ == "__main__":
    main()
