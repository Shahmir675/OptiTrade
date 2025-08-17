from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.dependencies import get_db
from app.models import pydantic_models as pyd_models
from app.models import sqlalchemy_models as sql_models
from app.services import email_service, security_service
from app.utils.avatar import select_random_avatar

router = APIRouter(prefix="", tags=["Authentication"])


@router.post("/signup")
async def signup(request: pyd_models.SignupRequest, db: AsyncSession = Depends(get_db)):
    if not all([request.u_name, request.email, request.u_pass]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="All fields are required."
        )

    query = select(sql_models.UserModel).where(
        sql_models.UserModel.email == request.email
    )
    result = await db.execute(query)
    if result.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already in use."
        )

    user_data = {
        "u_name": request.u_name,
        "email": request.email,
        "u_pass": request.u_pass,
    }
    await email_service.send_registration_otp(request.email, user_data)

    return {"message": "OTP sent. Please verify the OTP to complete registration."}


@router.post("/verify-otp")
async def verify_otp_route(
    request: pyd_models.OTPVerifyRequest, db: AsyncSession = Depends(get_db)
):
    is_valid, message, user_data_from_otp = security_service.verify_otp(
        request.email, request.otp
    )

    if not is_valid or not user_data_from_otp:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    security_service.clear_otp(request.email)

    hashed_password = security_service.hash_password(user_data_from_otp["u_pass"])
    new_user = sql_models.UserModel(
        u_name=user_data_from_otp["u_name"],
        email=user_data_from_otp["email"],
        u_pass=hashed_password,
        image_url=select_random_avatar(),
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    starting_balance = 10000.00
    new_user_balance = sql_models.UserBalanceModel(
        user_id=new_user.id,
        cash_balance=starting_balance,
        portfolio_value=0.00,
        net_worth=starting_balance,
    )
    db.add(new_user_balance)
    await db.commit()

    return {
        "message": "User registered successfully!",
        "user": {
            "id": new_user.id,
            "u_name": new_user.u_name,
            "email": new_user.email,
            "image_url": new_user.image_url,
        },
        "balance": {
            "cash_balance": starting_balance,
            "portfolio_value": 0.00,
            "net_worth": starting_balance,
        },
    }


@router.post("/login")
async def login(request: pyd_models.LoginRequest, db: AsyncSession = Depends(get_db)):
    query = select(sql_models.UserModel).where(
        sql_models.UserModel.email == request.email
    )
    result = await db.execute(query)
    user = result.scalars().first()

    if not user or not security_service.verify_password(request.u_pass, user.u_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token_data = {"id": user.id, "u_name": user.u_name, "email": user.email}
    token = security_service.create_access_token(data=token_data)

    return {
        "message": "Login successful!",
        "user": {"id": user.id, "u_name": user.u_name, "email": user.email},
        "token": token,
    }


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="Authorization")
    return {"message": "Logout successful!"}


@router.post("/forgot-password")
async def forgot_password(
    request: pyd_models.ForgotPasswordRequest, db: AsyncSession = Depends(get_db)
):
    query = select(sql_models.UserModel).where(
        sql_models.UserModel.email == request.email
    )
    result = await db.execute(query)
    if not result.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Email not found."
        )

    await email_service.send_password_reset_otp_email(request.email)
    return {"message": "OTP sent. Please verify the OTP to reset your password."}


@router.post("/verify-reset-otp")
async def verify_reset_otp(
    request: pyd_models.ResetPasswordRequest, db: AsyncSession = Depends(get_db)
):
    is_valid, message, _ = security_service.verify_otp(request.email, request.otp)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    user_result = await db.execute(
        select(sql_models.UserModel).where(sql_models.UserModel.email == request.email)
    )
    user = user_result.scalars().first()
    if not user:
        security_service.clear_otp(request.email)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )

    hashed_password = security_service.hash_password(request.new_password)
    user.u_pass = hashed_password
    db.add(user)
    await db.commit()

    security_service.clear_otp(request.email)
    return {"message": "Password successfully reset."}
